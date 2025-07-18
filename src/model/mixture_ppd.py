from typing import List, Dict, Union, Callable

import torch
from ifbo.transformer import TransformerModel
from model.calc_reliability import _calc_reliability
from model.mixing import EqualWeights

from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from ifbo import BarDistribution, FTPFN
from src.model.abstractmodel import AbstractModel
from src.model.batch_padded_pfn import MyBatch
from src.utils.filelogger_json import BufferedDictLogger
from utils.dotdict import DotDict


class PFNPPDMixture(AbstractModel):
    __name__ = "PFNPPDMixture"

    def __init__(
            self,
            model: Union[FTPFN, TransformerModel],
            logger: BufferedDictLogger,
            device,
            related_task_data: MyBatch,
            decay_fn: Callable,
            mixture_fn: Callable | None = None,
            reliability_fn: Callable = _calc_reliability,
            criterion=None,
            min_context_size: int = 10,
                    flippable_related=False

    ) -> None:
        """
        FT-PFN with Posterior Predictive Distribution Mixture.

        The idea is, that prior tasks are likely related. A FT-PFN, that is applied
        to the little data from the current task will initially have bad predictive performance.

        To leverage knowledge about the algorithms learning behaviour, we can improve the
        FT-surrogate, by considering the prior. In the case of the FT-PFN, we can build a
        Posterior Predictive mixture by:

        1. Computing the logits for the query points on the current task based on the current
        task's little data. These logits represent the in-context inferred posterior predictive
        distribution over the binned y distribution on the query. p_task(y^{task} | x^{task},
        D^{task - observed so far})
        2. For every related dataset at our disposal, we use that dataset's tokens to predict the
        same query points. Each will return logits, representing the respective belief over the
        posterior predictive distribution p_i(y^{task} | x^{task}, D^{related_task_i})
        3. We can now compute the mixture by weighing the contributions by their reliability;
        i.e. how likely the dataset is under the current data:
         p(y^{task} | x^{task}, \mathcal{D}) =
        \int_{D \in \mathcal{D}}p(y^{task} | x^{task}, D) p(D | D^{task - observed so far})

        Provided, that we know that some related tasks are in fact relevant, we may want to
        reduce variance and not trust the FT-PFN with too little data. That is why we he have a
        decayfactor function. It controls the relative mixture of the target logits with those of the related
        tasks over time; i.e. how much weight we give the prior given the current amount of
        target data.

        mixture_logits = alpha * target_logits + (1 - alpha) * related_logits

        :param model: pretrained FT-PFN Transformer
        :param logger:
        :param device:
        :param related_task_data:
        :param decayfactor: A parametrized function that will produce the current alpha
        :param mixture_fn: Callable, that will accept target_logits, related_logits,
        reliability_scores, alpha
        :param criterion: Bar distribution, that will allow us to interpret
        :param min_context_size:
        """

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
        self.flippable_related = flippable_related

    def query_batch_fwd(self, context_x, context_y, query_x):
        assert context_x.shape[0] + query_x.shape[0] <= 1000, \
            f"Distilled context + task context exceeds pfn's 1000 tokens. " \
            f"Distilled context: {context_x.shape[0]}, task context: {context_y.shape[0]}" \
            f"Cannot predict for query_x: {query_x.shape[0]}"

        T, B, num_bars = (query_x.shape[0], 1, self.criterion.num_bars)

        logits = torch.zeros((T, B, num_bars)).to(self.device)

        # ensuring that if we can do the entire query in one go, we do that
        query_size = min(1000 - context_x.shape[0], query_x.shape[0])
        for t in range(0, query_x.shape[0], query_size):
            logits[t:t + query_size, :, :] = self.model(
                (
                    torch.cat([context_x, query_x[t:t + query_size]], dim=0),
                    context_y
                ),
                single_eval_pos=context_x.shape[0]
            )

        return logits

    def _forward(self, context_x, context_y, query_x, minimize=False, *args, **kwargs) -> (
            torch.Tensor):
        """
        Posterior Predictive Mixture based on the reliability of the related tasks
        and decaying the prior's importance based on the context size of the current task.

        1. We attempt to first predict the reliability of a task, by predicting
        the likelihood of the context points under the related tasks.

        2. We then use the reliability scores to weight the logits of the related tasks
        for the query points (which is their respective PPD)

        Parameters:
            :param context_x: The context points of the current task.
            :param context_y: The context values of the current task.
            :param query_x: The query points for the current task.
            :return: The mixed logits for the query points weighting the prior derived
            from the related tasks with the logits based of the observed part of the current task.
        """

        # (1) calculate the likelihood of the context points under the related tasks
        # Consider: reference NLL could be nll on leave one out from the task (which can
        #  efficiently be calculated in a batch with the same separation!).
        query_x = query_x.to(self.device)
        context_size = context_x.shape[0]
        T, n_related_tasks, num_bars = (
            query_x.shape[0],
            len(self.related_task_data),
            self.criterion.num_bars
        )

        context_x = context_x.to(self.device)
        context_y = context_y.to(self.device)

        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)


        related_task_data = DotDict({
            'x': related_context_x,
            'y': related_context_y,
            'padding_mask': padding_mask,
            'single_eval_pos': related_context_x.shape[0]
        })

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
            reliability_scores = -torch.log(torch.ones(len(self.related_task_data)))

        reliability_scores = reliability_scores.to(self.device)

        # 0 idx is reserved for the current task
        for i, score in enumerate(reliability_scores):
            self.logger.log({'reliability_score': score.item(), 'task_index': i, 'context_size': context_x.shape[0],})


        # (3) TODO Localize the query_points; i.e. first find the most relevant context points
        #       for each task, then potentially fill it up with the rest of the context points
        #       available

        # CONSIDER: instead of back-propping to find local points via distillation, we could also
        #  just query the PFN at the position of the query directly and sample from the distribution.
        # Consider: knn localization --> ablate over whether we need additional context other
        #  than knn samples

        # (2) Collect the PPD logits on the target task's query

        # related ppd for current query points
        task_context_x = related_task_data.x.to(self.device)
        task_context_y = related_task_data.y.to(self.device)
        padding_mask = related_task_data.padding_mask.to(self.device) if isinstance(related_task_data, MyBatch) else None
        num_related = task_context_x.shape[1]


        related_logits = self.model(
            (
                torch.cat([task_context_x, query_x.repeat(1, num_related, 1)], dim=0),
                task_context_y
            ),
            single_eval_pos=task_context_x.shape[0],
            src_key_padding_mask=padding_mask
        )

        target_logits = self.model(
            (
                torch.cat([context_x, query_x], dim=0),
                context_y
            ),
            single_eval_pos=context_x.shape[0],
            src_key_padding_mask=None
        )

        #   # (4) Calculate the mixture of the logits based on the reliability scores

        # mixture_logits = self.mixture_fn(
        #     target_logits,
        #     related_logits,
        #     reliability_scores,
        #     # temperature=temperature,
        #     alpha=self.decay_fn(context_size)
        # )

        # if self.mixture_fn is None:
        logits = torch.cat([target_logits, related_logits], dim=1)
        n = logits.shape[1]
        mixture_logits = (logits * torch.ones((T, n, 1)).to(self.device) / n).sum(dim=1, keepdim=True)

        return mixture_logits

    @torch.no_grad()
    def get_pi(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        logits = self(x_train=x_train, y_train=y_train, x_test=x_test, minimize=minimize)
        # torch.Size([x_train.shape[0], 1, 10000])
        scores = self.criterion.pi(logits.squeeze(), best_f=inc)
        return scores


