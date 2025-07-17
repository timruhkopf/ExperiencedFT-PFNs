from pathlib import Path

import torch

from ifbo.transformer import TransformerModel
from src.model.abstractmodel import AbstractModel
from src.model.calc_reliability import _calc_reliability

import logging

from src.utils.dotdict import DotDict
from utils.filelogger import BufferedFileLogger

log = logging.getLogger(__name__)
import tensorboard


class PFNPriorImputation(AbstractModel):
    """
    Prior-Fitted-Netowrk with Prior Imputation on observed data from the target task
    to find interesting acqusition points

    Core Idea:
    ----------
    Rather than translating experience between tasks, we use the prior(s) to retrospectively evaluate
    the collected data points. For each prior, we ask: "Given only my prior beliefs, what acquisition
    values would I assign to the observed points?" This provides a set of prior-informed acquisition
    landscapes, which can be combined with the target-task's acquisition values to steer exploration
    toward regions that are promising under both prior and empirical evidence.

    Workflow:
    ---------
    1. **Target Task:** Compute the acquisition function (Probability of Improvement, PI variant) for $x_q$
       using the posterior predictive distribution (PPD) conditioned on the current target history $\tau^*_{1:b}$:

           $$p(y | x_q, \tau^*_{1:b})$$

    2. **Prior Tasks:** For each prior $\tau_i$:
        - Predict the PPD for the *observed* points $x \in \tau^*_{1:b}$ under the prior:

              $$p(y | x \in \tau^*_{1:b}, \tau_i)$$

        - Evaluate the PI acquisition function at these points under the prior.
        - Optionally, impute $\hat{y}$ for these points under the prior and augment the prior history
          (for self-confirmation analysis, but not required for the main workflow).
        - Predict the PPD for $x_q$ under the prior (with or without imputation).
        - Evaluate the PI acquisition function for $x_q$ under the prior.

    3. **Aggregation:**
        - Aggregate the acquisition values from all priors and the target task.
        - Aggregation strategies may include:
            - **Averaging:** Simple mean of acquisition values across priors and target.
            - **Weighted Averaging:** Weight each prior’s contribution by relevance, confidence, or empirical fit (e.g., likelihood of prior on observed data).
            - **Ensemble Selection:** Use a voting or ranking scheme to select the most promising $x_q$.
            - **Stacked Acquisition:** Learn a meta-model to combine acquisition values (advanced).

    4. **Selection:**
        - Use the aggregated acquisition values to select the next $x_q$ for evaluation.

    Notes:
    ------
    - The intent is to let the prior(s) steer the search, especially in low-data regimes, by
      leveraging their “opinion” of the search space.
    - The PI variant is used as the acquisition function throughout.
    """
    __name__ = "PFNPriorImputation"

    def __init__(self, model, criterion, logger, mixture_strategy,
                 related_task_data, min_context_size, imputation_mode='mean',
                 incumbent_calculation='imputation only', flippable_related=False,
                 device=None, verbose=True):
        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model

        # if criterion is not None:
        #     self.criterion = criterion
        # elif isinstance(model, FTPFN):
        #     self.criterion = model.model.criterion
        # elif isinstance(model, TransformerModel):
        #     self.criterion = model.criterion
        self.criterion = criterion if criterion is not None else self.model.criterion
        self.logger = logger
        self.device = device

        self.related_task_data = related_task_data

        self.min_context_size = min_context_size
        self.imputation_mode = imputation_mode
        self.flippable_related = flippable_related

        self.mixture_strategy = mixture_strategy
        self.call_counter = 0
        self.verbose = verbose
        self.incumbent_calculation = incumbent_calculation

        if verbose:
            num_related = self.related_task_data.x.shape[1]
            self.mixture_logger = BufferedFileLogger(
                file_name=f"mixture_strategy.csv",
                file_path=Path().cwd() / "logs" / "mixture_strategy",
                header=['metric', 'step', 'target_reliability',
                        *[f'related_reliability_{i}' for i in range(num_related)]],
                postfix=[]
            )

    def _forward(self, context_x, context_y, query_x, *args, **kwargs) -> torch.Tensor:

        context_x = context_x.to(self.device)
        context_y = context_y.to(self.device)
        query_x = query_x.to(self.device)

        # Target task logits ---------------------------------------------------
        target_logits = self.model(
            (
                torch.cat([context_x, query_x], dim=0),
                context_y
            ),
            single_eval_pos=context_x.shape[0],
            src_key_padding_mask=None
        )

        return target_logits

    def calculate_reliability(self, x_train, y_train):
        context_size = x_train.shape[0]

        context_x = x_train.to(self.device)
        context_y = y_train.to(self.device)

        related_x = self.related_task_data.x.to(self.device)
        related_y = self.related_task_data.y.to(self.device)
        padding_mask = self.related_task_data.padding_mask.to(self.device)
        single_eval_pos = related_x.shape[0]

        related_task_data = DotDict({
            'x': related_x,
            'y': related_y,
            'padding_mask': padding_mask,
            'single_eval_pos': single_eval_pos
        })

        # fixme: cache these values when we move to optimizing the acquisition function
        if context_size >= self.min_context_size:
            # calculate the reliability scores for the related tasks
            counter = self.call_counter + 1 - self.min_context_size
            reliability_scores = self.reliability_fn(
                self.model,
                context_x,
                context_y,
                related_task_data,
                self.criterion,
                verbose=True if self.verbose and counter < 15 or counter % 100 == 0 else False,
                plot_file_path=Path().cwd() / f"reliability_scores_{counter}.png"
            )
        else:
            # if we have no context points, we must assume all are equally likely
            reliability_scores = -torch.log(torch.ones(len(self.related_task_data),
                                                       device=self.device))

        # 0 idx is reserved for the current task
        for i, score in enumerate(reliability_scores):
            self.logger.add_scalar(
                "reliability_score",
                score.item(),
                -1,  # step
                context_x.shape[0],
                i,  # task index
            )

        return reliability_scores

    def impute(self, x_train: torch.Tensor, task_context_x, task_context_y, padding_mask) -> (
            torch.Tensor):
        """
        Impute the y values for the training data from the target task under the prior context.
        """
        # related ppd for current query points
        num_related = task_context_x.shape[1]

        if task_context_x.shape[1:] != x_train.repeat(1, num_related, 1).shape[1:]:
            print(f"Shape mismatch: {task_context_x.shape[1:]} vs"
                  f"{x_train.repeat(1, num_related, 1).shape[1:]}")

        # impute the observed data points --------------------------------------
        # TODO Cache these values, when we optimize over the acquisition function?
        imputed_logits = self.model(
            (
                # fixme: slurm error: ifbo_array_5565440_5.err
                #  RuntimeError: Sizes of tensors must match except in dimension 0. Expected size
                #  10 but got size 6 for tensor number 1 in the list.
                torch.cat([task_context_x, x_train.repeat(1, num_related, 1)], dim=0),
                torch.cat([task_context_y, ], dim=0)
            ),
            single_eval_pos=task_context_x.shape[0],
            src_key_padding_mask=padding_mask
        )

        if self.imputation_mode == 'median':
            imputed_y = self.criterion.median(imputed_logits)
        elif self.imputation_mode == 'mean':
            imputed_y = self.criterion.mean(imputed_logits)
        elif self.imputation_mode == 'sample':
            # Sample indices from the categorical distributions
            # Shape: (T, n_related_tasks, 1)
            probs = imputed_logits.softmax(-1)
            bins = self.criterion.borders

            # to get the sample at the middle of the bins, we can calculate the middle points
            bucket_middle = (bins[:-1] + bins[:-1] + self.criterion.bucket_widths) / 2

            sampled_indices = torch.stack([
                torch.multinomial(probs[:, i, :], 1) for i in range(probs.shape[1])
            ], dim=1)

            # Remove the last dimension for direct indexing
            # Shape: (T, n_related_tasks)
            sampled_indices = sampled_indices.squeeze(-1)

            # Gather the corresponding bin values
            # Shape: (T, n_related_tasks, num_bars) if bins is 2D, else (T, n_related_tasks)
            imputed_y = bucket_middle[sampled_indices]
        else:
            raise ValueError(f"Unknown imputation mode: {self.imputation_mode}")

        return imputed_y

    def get_pi_related(self, x_train, x_test, related_context_x=None, related_context_y=None,
                       padding_mask=None):
        """
        First impute the y values for the target task under the related prior context,
        then calculate the incumbent under the imputed data and finally collect the
        pi values for the query points under the imputed prior.

        :param x_train: target task training data points (x)
        :param x_test: query points for which to calculate the pi values
        :param related_context_x:
        :param related_context_y:
        :param padding_mask: padding for the related context data
        :return:
        """
        num_related = related_context_x.shape[1]

        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        related_context_x = related_context_x.to(self.device)
        related_context_y = related_context_y.to(self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        imputed_y = self.impute(
            x_train.unsqueeze(1),
            related_context_x,
            related_context_y,
            padding_mask=padding_mask
        )

        # prior logits under the imputed data points --------------------------

        # 2.
        # TODO key-value-cache here on related tasks and incremental x_train
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.unsqueeze(1).repeat(1, num_related, 1),
                    x_test.unsqueeze(1).repeat(1, num_related, 1)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            src_key_padding_mask=torch.cat([
                padding_mask,
                torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            ], dim=1)
        )

        B = prior_logits.shape[1]
        # TODO ABLATE consider that we should only use the imputed_y here for incumbent calculation.
        if self.incumbent_calculation == 'imputation only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related only':
            prior_incumbents = related_context_y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation and related':
            prior_incumbents = torch.cat([related_context_y, imputed_y, ], dim=0).max(dim=0).values
        else:
            raise ValueError(f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])
        pi_related = torch.stack([
            self.criterion.pi(prior_logits[:, b, :].squeeze(1),
                              best_f=prior_incumbents[b, :].unsqueeze(1),
                              maximize=True)  # FIXME: do we need to maximize here?
            for b in range(B)
        ], dim=0)

        return pi_related

    def get_pi_target(self, x_train, y_train, x_test, inc):
        """
        Calculate the Probability of Improvement (PI) acquisition function for the
        query points under the target task.
        :param x_train:
        :param y_train:
        :param x_test:
        :param inc: incumbent under the target task
        :return:
        """
        target_logits = self.model(
            (
                torch.cat([x_train.unsqueeze(1), x_test.unsqueeze(1)], dim=0),
                y_train.unsqueeze(1)
            ),
            single_eval_pos=x_train.shape[0],

        )
        pi_target = self.model.criterion.pi(
            target_logits.squeeze(1), best_f=inc,
            maximize=True)  # FIXME: do we need to maximize here?
        return pi_target

    @torch.no_grad()
    def get_pi(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        """
        Get the Probability of Improvement (PI) acquisition function for the
        query points under the target task and the related tasks.

        1. Impute the y values for the training data from the target task under the prior context.
        2. Calculate the prior logits for the imputed data points and the test (query) points, then
        calculate the acquisition values for the query points under the imputed prior.
        3. Calculate the acquisition values for the query points under the target task.
        4. Weigh the acquisition values by the reliability scores and prior decay.

        :param x_test: The query points for which to calculate the acquisition function.
        :param inc: The current best observed value (incumbent) for the target task.
        :param x_train: The training data points for the target task.
        :param y_train: The training labels for the target task.
        """
        self.call_counter += 1
        # 1.
        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask

        x_test = x_test.to(self.device)
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        inc = inc.to(self.device)

        if torch.any(x_test > 999.):
            log.warning(f"Query points x_test contain values > 999: {x_test[x_test > 999.]}")
            x_test = torch.clamp(x_test, max=999.)
        if torch.any(x_train > 999.):
            log.warning(f"Training points x_train contain values > 999: {x_train[x_train > 999.]}")
            x_train = torch.clamp(x_train, max=999.)

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)

        related_task_data = DotDict({
            'x': related_context_x,
            'y': related_context_y,
            'padding_mask': padding_mask,
            'single_eval_pos': related_context_x.shape[0]
        })

        pi_related = self.get_pi_related(
            x_train=x_train,
            x_test=x_test,
            related_context_x=related_context_x,
            related_context_y=related_context_y,
            padding_mask=padding_mask
        )
        # 3 # FIXME: with proper stacking, the prior and target logits could be calculated in one go
        #      this implementation here is just to keep the code simple and readable for debugging
        pi_target = self.get_pi_target(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            inc=inc
        )

        # Plotting the reliability scores for debugging
        if self.verbose:
            # df = self.mixture_logger.dataframe
            # plot_reliability_scores(df)

            # measure the correlation between the target and mean of the related pi values
            # Step 1: Compute mean of related pi values
            pi_related_mean = pi_related.mean(dim=0)  # shape [101]

            # Step 2: Calculate Pearson correlation coefficient
            stacked = torch.stack([pi_target, pi_related_mean], dim=0)  # [2, 101]
            corr_matrix = torch.corrcoef(stacked)  # [2, 2]
            corr_value = corr_matrix[0, 1].item()  # scalar Pearson correlation

            # Step 3: Log it
            self.mixture_logger.add_scalar(
                "pi/corr_target_related_mean",
                self.call_counter,
                corr_value
            )
            # plot_pi_correlation(self.mixture_logger.dataframe)

            # plot_pi(pi_target, pi_related)

        # 4.
        scores = self.mixture_strategy(
            model=self.model,
            criterion=self.criterion,
            related_task_data=related_task_data,
            logger=self.mixture_logger if self.verbose else None,
        )(
            x_train=x_train,
            y_train=y_train,
            pi_target=pi_target,
            pi_related=pi_related,
        )


        return scores


def plot_pi(pi_target, pi_related):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(15, 5))

    # Boxplot (one box per index)
    ax.boxplot(pi_related.T, positions=np.arange(pi_related.shape[1]), widths=0.6,
               patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Overlay: pi_target as a red dot for each index
    ax.scatter(np.arange(pi_target.shape[0]), pi_target, color='red', zorder=10, s=20,
               label='pi_target')

    ax.set_xticks(np.arange(0, pi_target.shape[0], 10))
    ax.set_xlabel('Index')
    ax.set_ylabel('pi value')
    ax.set_title('pi_related (boxplot) & pi_target (red dot) per index')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_reliability_scores(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Filter rows where metric == 'reliability'
    df_rel = df[df['metric'] == 'reliability']
    # Select columns to plot
    cols_to_plot = ['target_reliability'] + [col for col in df.columns if
                                             col.startswith('related_reliability_')]
    # Melt dataframe to long format for seaborn
    df_long = df_rel.melt(id_vars='step', value_vars=cols_to_plot,
                          var_name='reliability_type', value_name='reliability_value')
    # Plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_long, x='step', y='reliability_value', hue='reliability_type')
    plt.title('Weight vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Softmax weight')
    plt.show()


def plot_pi_correlation(df):
    # FIXME: something went wrong with the logging --> the columns here are messed up
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Filter rows for each metric
    df_corr = df[df['metric'] == 'pi/corr_target_related_mean']
    df_rel = df[df['metric'] == 'reliability']

    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()  # Primary axis

    # Plot first metric on primary y-axis (left)
    sns.lineplot(data=df_corr, x='step', y='target_reliability', marker='o', ax=ax1,
                 label='Corr(pi_target, mean(pi_related))')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_xlabel('Step')
    ax1.axhline(y=0, color='blue', )

    # Create secondary y-axis (right)
    ax2 = ax1.twinx()
    sns.lineplot(data=df_rel, x='step', y='target_reliability', marker='s', ax=ax2, color='orange',
                 label='target weight')
    ax2.set_ylabel('Reliability')
    ax2.axhline(y=0.2, color='orange', )


    # Titles and grid
    plt.title('Correlation and Reliability over Steps')
    ax1.grid()
    # Optional: handle legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.show()
