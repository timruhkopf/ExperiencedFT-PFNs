import torch

from ifbo.transformer import TransformerModel
from src.model.abstractmodel import AbstractModel
from src.model.calc_reliability import _calc_reliability

import logging

from src.utils.dotdict import DotDict

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

    def __init__(self, model, criterion, logger, decay_fn, mixture_fn,
                 related_task_data, min_context_size, imputation_mode='mean',
                 reliability_fn=_calc_reliability,
                 device=None):
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

        self.reliability_fn = reliability_fn
        self.decay_fn = decay_fn
        self.min_context_size = min_context_size
        self.imputation_mode = imputation_mode

        self.mixture_fn = mixture_fn
        self.call_counter=0
        # TODO: distill the related tasks once (optionally)

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
            reliability_scores = self.reliability_fn(
                self.model,
                context_x,
                context_y,
                related_task_data,
                self.criterion,
                # peeking={
                # # this was just to check why the reliability scores were so little
                # # indicative of the actual performance in the first iteration.
                #     'x': context_x,
                #     'y': context_y
                # }
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
        scores_related = torch.stack([
            self.criterion.pi(prior_logits[:, b, :].squeeze(1), best_f=inc)
            for b in range(B)
        ], dim=0)

        # 3 # FIXME: with proper stacking, the prior and target logits could be calculated in one go
        #      this implementation here is just to keep the code simple and readable for debugging
        target_logits = self.model(
            (
                torch.cat([x_train.unsqueeze(1), x_test.unsqueeze(1)], dim=0),
                y_train.unsqueeze(1)
            ),
            single_eval_pos=x_train.shape[0],

        )
        scores = self.model.criterion.pi(target_logits.squeeze(1), best_f=inc)

        # 4.
        scores = self.mixture_fn(
            scores,
            scores_related,
            reliability_scores=self.calculate_reliability(
                x_train.unsqueeze(1), y_train.unsqueeze(1),
                minimize=minimize
            ).to(self.device),
            alpha=self.decay_fn(x_train.shape[0])
        )
        self.call_counter += 1
        return scores
