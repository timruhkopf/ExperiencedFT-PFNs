from pathlib import Path

import torch

from ifbo.transformer import TransformerModel
from src.model.abstractmodel import AbstractModel
from src.model.calc_reliability import _calc_reliability

import logging

from src.utils.dotdict import DotDict
from src.utils.filelogger import BufferedFileLogger
from src.model.utils import general_power_transform

log = logging.getLogger(__name__)


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
                 related_task_data, min_context_size, imputation_mode='mean',only_obs_incumbents = True,
                 device=None, verbose=False):

        if "MPFNs4BO" in type(model).__name__ or "ourPFNs4BO" in type(model).__name__:
            self.model = model
        else:
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

        self.reliability_scores = None 

        self.min_context_size = min_context_size
        self.imputation_mode = imputation_mode
        self.only_obs_incumbents = only_obs_incumbents  
        self.call_counter = 0
        self.verbose = verbose

        if verbose:
            num_related = self.related_task_data.x.shape[1]
            self.mixture_logger = BufferedFileLogger(
                file_name=f"mixture_strategy.csv",
                file_path=Path().cwd() / "logs" / "mixture_strategy",
                header=['metric', 'step', 'target_reliability',
                        *[f'related_reliability_{i}' for i in range(num_related)]],
                postfix=[]
            )

        self.mixture_strategy =  mixture_strategy(
            model=self.model,
            criterion=self.criterion,
            related_task_data=related_task_data,
            logger=self.mixture_logger if self.verbose else None,
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

    def calculate_reliability(self, x_train, y_train, minimize=False):
        context_size = x_train.shape[0]

        context_x = x_train.to(self.device)
        context_y = y_train.to(self.device)

        related_x = self.related_task_data.x.to(self.device)
        related_y = self.related_task_data.y.to(self.device)
        padding_mask = self.related_task_data.padding_mask.to(self.device)
        single_eval_pos = related_x.shape[0]

        if minimize:
            related_y = (1 - related_y)

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
        if self.logger is not None:
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

        # impute the observed data points --------------------------------------
        # TODO Cache these values, when we optimize over the acquisition function?
        imputed_logits = self.model(
            (
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
                       padding_mask=None,minimize = False, apply_power_transform=False):
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

        imputed_y = self.impute(
            x_train.unsqueeze(1),
            related_context_x,
            related_context_y,
            padding_mask=padding_mask
        )

        self.imputed_y = imputed_y

        # prior logits under the imputed data points --------------------------

        # 2.
        # TODO key-value-cache here on related tasks and incremental x_train

        y_values = torch.cat([related_context_y, imputed_y, ], dim=0)
        if apply_power_transform:
            y_values = general_power_transform(y_values, y_values)
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.unsqueeze(1).repeat(1, num_related, 1),
                    x_test.unsqueeze(1).repeat(1, num_related, 1)
                ], dim=0),
                y_values
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            src_key_padding_mask=torch.cat([
                padding_mask,
                torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            ], dim=1)
        )

        B = prior_logits.shape[1]

        if self.only_obs_incumbents:
            incumbents = y_values[related_context_y.shape[0]:]
        else: # the problem is PI gets clipped if we use only the imputed_y
            incumbents = y_values
        
        if minimize:
            prior_incumbents = incumbents.min(dim=0).values
        else:
            prior_incumbents = incumbents.max(dim=0).values
        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])

        
        pi_related = torch.stack([
            self.criterion.pi(prior_logits[:, b, :].squeeze(1),
                            best_f=prior_incumbents[b, :])
            for b in range(B)
        ], dim=0)

        return pi_related

    def get_pi_target(self, x_train, y_train, x_test, inc, minimize=False, apply_power_transform=False):
        """
        Calculate the Probability of Improvement (PI) acquisition function for the
        query points under the target task.
        :param x_train:
        :param y_train:
        :param x_test:
        :param inc: incumbent under the target task
        :return:
        """

        if apply_power_transform:
            y = general_power_transform(y_train.unsqueeze(1), y_train.unsqueeze(1))
            inc = y.min() if minimize else y.max()
        else:
            y = y_train.unsqueeze(1)
        target_logits = self.model(
            (
                torch.cat([x_train.unsqueeze(1), x_test.unsqueeze(1)], dim=0),
                y
            ),
            single_eval_pos=x_train.shape[0],

        )
        pi_target = self.model.criterion.pi(target_logits.squeeze(1), best_f=inc, maximize = not minimize)
        return pi_target

    @torch.no_grad()
    def get_pi(self, x_test, inc, x_train=None, y_train=None, minimize=False, apply_power_transform=False):
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
        # 1.
        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask
        num_related = related_context_x.shape[1]
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

        if minimize:
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
            padding_mask=padding_mask,
            minimize = minimize,
            apply_power_transform=apply_power_transform
        )
        # 3 # FIXME: with proper stacking, the prior and target logits could be calculated in one go
        #      this implementation here is just to keep the code simple and readable for debugging
        pi_target = self.get_pi_target(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            inc=inc,
            minimize=minimize,
            apply_power_transform=apply_power_transform
        )
        self.pi_target = pi_target
        self.pi_related = pi_related

        # 4.
        scores, reliability_scores =self.mixture_strategy(
            x_train=x_train,
            y_train=y_train,
            pi_target=pi_target,
            pi_related=pi_related,
            minimize=minimize,  # fixme: do we need this?
        )
        self.reliability_scores = reliability_scores  
        #print(f"Reliability scores: {self.reliability_scores}")
        self.call_counter += 1
        return scores
