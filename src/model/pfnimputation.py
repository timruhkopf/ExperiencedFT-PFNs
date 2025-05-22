import torch

from ifbo.transformer import TransformerModel
from model.abstractmodel import AbstractModel


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
                 reliability_fn, related_task_data, min_context_size, device=None):
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

        self.mixture_fn = mixture_fn

    def _forward(self, context_x, context_y, query_x, *args, **kwargs) -> torch.Tensor:
        context_size = context_x.shape[0]
        T, n_related_tasks, num_bars = (
            query_x.shape[0],
            len(self.related_task_data),
            self.criterion.num_bars
        )

        context_x = context_x.to(self.device)
        context_y = context_y.to(self.device)

        # fixme: cache these values when we move to optimizing the acquisition function
        if context_size >= self.min_context_size:
            # calculate the reliability scores for the related tasks
            reliability_scores = self.reliability_fn(
                self.model,
                context_x,
                context_y,
                self.related_task_data,
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
            reliability_scores = -torch.log(torch.ones(len(self.related_task_data)))

        # 0 idx is reserved for the current task
        for i, score in enumerate(reliability_scores):
            self.logger.add_scalar(
                "reliability_score",
                score.item(),
                -1,  # step
                context_x.shape[0],
                i,  # task index
            )

        # related ppd for current query points
        task_context_x = self.related_task_data.x
        task_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask
        num_related = task_context_x.shape[1]

        # impute the observed data points --------------------------------------
        # TODO Cache these values, when we optimize over the acquisition function?
        imputed_logits = self.model(
            (
                torch.cat([task_context_x, context_x.repeat(1, num_related, 1)], dim=0),
                torch.cat([task_context_y, ], dim=0)
            ),
            single_eval_pos=task_context_x.shape[0],
            src_key_padding_mask=padding_mask
        )

        imputations = self.criterion.median(imputed_logits)

        # prior logits under the imputed data points --------------------------
        prior_logits = self.model(
            (
                torch.cat([task_context_x, context_x.repeat(1, num_related, 1), query_x.repeat(1, num_related, 1)], dim=0),
                torch.cat([task_context_y, imputations, ], dim=0)
            ),
            single_eval_pos=task_context_x.shape[0] + context_x.shape[0],
            src_key_padding_mask=padding_mask
        )

        # Target task logits ---------------------------------------------------
        target_logits = self.model(
            (
                torch.cat([context_x, query_x], dim=0),
                context_y
            ),
            single_eval_pos=context_x.shape[0],
            src_key_padding_mask=None
        )