from typing import List, Dict, Union

import torch
from ifbo.transformer import TransformerModel

from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.ifBO_main.ifbo import BarDistribution, FTPFN
from src.model.abstractmodel import AbstractModel


def _calc_reliability(
        model: TransformerModel,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        related_task_data: List[Dict[str, torch.Tensor]],  # type: ignore
        criterion: BarDistribution
) -> torch.Tensor:
    """
    Calculate the reliability of the related task with respect to the current task.

    Basically we try to infer the likelihood of the task's observed points under the related task.

    Args:
        :param model: The model to use for the calculation.
        :param context_x: The context points of the current task.
        :param context_y: The context values of the current task.
        :param related_task_data: The data of the related tasks.
        :param criterion: The criterion to use for the calculation.

    """
    reliability_scores = []
    for task_data in related_task_data:
        task_context_x = task_data['x']
        task_context_y = task_data['y']

        logits = model(
            (
                torch.cat([task_context_x, context_x], dim=0),
                task_context_y
            ),
            single_eval_pos=task_context_x.shape[0]
        )
        # y's associated with query for that task
        target = context_y
        loss = criterion(logits, target)
        loss = loss.view(-1, logits.shape[1])  # bar distribution issue
        reliability_scores.append(loss.mean().item())

    return torch.tensor(reliability_scores).to(context_x.device)


class PFNPPDMixture(AbstractModel):
    def __init__(
            self,
            model: Union[FTPFN, TransformerModel],
            logger,
            device,
            related_task_data: List[Dict[str, torch.Tensor]],
            criterion=None,
            decayfactor=lambda x: 1.0,
            min_context_size: int = 10
    ) -> None:
        self.min_context_size = min_context_size

        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model
        self.model = self.model.to(device)
        self.criterion = criterion if criterion is not None else model.criterion
        self.logger = logger
        self.device = device

        self.related_task_data = related_task_data
        self.decay_factor = decayfactor

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

    def forward(self, *args, **kwargs):
        """
        Interface for the TransformerModel class from ifbo.
        """
        if isinstance(args, tuple) and len(args) == 1:
            # this is the unfortunate TransformerModel compatability
            args = args[0]
            x, y = args
            single_eval_pos = kwargs['single_eval_pos']
            kwargs = {
                'context_x': x[:single_eval_pos],
                'context_y': y,
                'query_x': x[single_eval_pos:]
            }

        if 'single_eval_pos' in kwargs:
            del kwargs['single_eval_pos']

        return self._forward(**kwargs)

    @torch.no_grad()
    def _forward(self, context_x, context_y, query_x, temperature=1) -> (
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
            :param temperature: The temperature for the softmax calculation.
            Smaller values will give more weight to higher reliability scores,
            i.e. we will rely less on unreliable experiences.
            :return: The mixed logits for the query points weighting the prior derived
            from the related tasks with the logits based of the observed part of the current task.
        """

        # (1) calculate the likelihood of the context points under the related tasks
        # Consider: reference NLL could be nll on leave one out from the task (which can
        #  efficiently be calculated in a batch with the same separation!).

        context_size = context_x.shape[0]
        T, n_related_tasks, num_bars = (
            query_x.shape[0],
            len(self.related_task_data),
            self.criterion.num_bars
        )

        if context_size >= self.min_context_size:
            # calculate the reliability scores for the related tasks
            reliability_scores = _calc_reliability(
                self.model,
                context_x,
                context_y,
                self.related_task_data,
                self.criterion
            )
        else:
            # if we have no context points, we must assume all are equally likely
            reliability_scores = -torch.log(torch.ones(len(self.related_task_data))).to(
                context_x.device)

        # 0 idx is reserved for the current task
        for i, score in enumerate(reliability_scores, start=1):
            self.logger.add_scalar(
                "reliability_score",
                score.item(),
                i,  # task index
                context_x.shape[0]
            )

        # (3) TODO Localize the query_points; i.e. first find the most relevant context points
        #       for each task, then potentially fill it up with the rest of the context points
        #       available

        # CONSIDER: instead of back-propping to find local points via distillation, we could also
        #  just query the PFN at the position of the query directly and sample from the distribution.
        # Consider: knn localization --> ablate over whether we need additional context other
        #  than knn samples

        # (2) Collect the PPD logits
        logits = torch.zeros(T, 1 + n_related_tasks, num_bars)

        if context_size >= self.min_context_size:
            task = {'x': context_x, 'y': context_y}
            all_tasks = [task, *self.related_task_data]
        else:
            all_tasks = self.related_task_data

        # FIXME: make this a single padded batch fwd
        for b, task_data in enumerate(all_tasks):
            task_context_x = task_data['x']
            task_context_y = task_data['y']
            logits[:, b:b + 1, :] = self.query_batch_fwd(
                task_context_x,
                task_context_y,
                query_x
            )

        if context_size >= self.min_context_size:
            # add uniform logits for the current task
            logits[:, 0:1, :] = torch.zeros((T, 1, num_bars)).to(self.device)

        # get the mixture distribution
        mixed = self.calc_logits_mixture(
            logits,
            reliability_scores,
            context_size,
            temperature
        )

        return mixed

    def calc_logits_mixture(
            self,
            logits: torch.Tensor,
            reliability_scores,
            context_size: int,
            temperature=1.
    ) -> torch.Tensor:
        """
        Calculate the mixture of the logits based on the reliability scores.

        :param logits: T, B+1, C, with B+1 being the number of tasks (including the current task,
        located at index 0)
        :param reliability_scores: nll scores for each task
        :param context_size: size of the context for the current task
        :param temperature: temperature for the softmax calculation over the reliability scores.
        # FIxME: temperature is an important hyperparameter, as lower values will
        #  give more weight to higher reliability scores (fewer tasks will be
        #  considered)
        :return: logits tensor T, 1, C, which defines the mixture of the logits
        """
        if reliability_scores.shape[0] > 1:
            weights = torch.softmax(-reliability_scores / temperature)
        else:
            weights = torch.tensor([1.0])

        weights = weights.to(self.device)

        if context_size > self.min_context_size:
            # devalue the current task
            weights[0] *= self.decay_factor(context_size)

        weights = torch.cat([torch.tensor([1.0]).to(self.device), weights])
        weights = weights / weights.sum()  # normalizing with the current task

        for i, score in enumerate(weights.to('cpu')):
            self.logger.add_scalar(
                "weights",
                score.item(),
                i,  # task index
                context_size
            )

        logits = (logits * weights.view(1, 2, 1)).sum(dim=1)

        # Weighted combination using broadcasting
        return logits.unsqueeze(1)  # [B, 1, C]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    from src.utils.filelogger import BufferedFileLogger
    import matplotlib.pyplot as plt

    # FIXME: factor this into a set of fixtures
    import ifbo
    from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch
    import tempfile

    BATCH_SIZE = 64
    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 50  # FIXME: important hyperparameter
    N_STEPS = 10  # number of distillation steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=2, single_eval_pos=500)

    sep = batch.single_eval_pos[0]
    x, y = batch.x, batch.y

    # 0 task is the one we want to predict on
    target_task_context_x = x[:sep, 0:1]
    target_task_context_y = y[:sep, 0:1]
    target_task_val_x = x[sep:, 0:1]
    target_task_val_y = y[sep:, 0:1]

    # fixme: max context size is 1k, so distilled + task context + query can
    #  exceed that. in that case we will want to batch over the exceeing query points with the
    #  same context.
    max_query_size = 1000 - target_task_context_x.shape[0] - DISTILL_SIZE
    target_task_query_x = x[sep:sep + max_query_size, 0:1]  # actual test points
    target_task_query_y = y[sep:sep + max_query_size, 0:1]  # actual test labels

    x_task_context = x[:sep, 1:]
    y_task_context = y[:sep, 1:]
    x_val = x[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    y_val = y[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    # print(x.shape, y.shape)

    target_task_context = {
        'x': target_task_context_x,
        'y': target_task_context_y
    }

    target_task_query = {
        'x': target_task_query_x,
        'y': target_task_query_y
    }

    related_task_data = [{
        'x': x_task_context,
        'y': y_task_context
    }]


    with tempfile.TemporaryDirectory() as tmpdirname:
        trainlogger = BufferedFileLogger(
            file_name="distill.csv",
            file_path=tmpdirname,
            buffer_size=1000,
            header=("metric", "value", "task", "context_size"))

        pfnmixture = PFNPPDMixture(
            pfn_backend,
            trainlogger,
            device,
            related_task_data,
            criterion=pfn_backend.criterion,
            min_context_size=10,
            # exponential decay factor 1 for size of 10, 0 for 200
            # decayfactor=lambda size: 1 - (size - 10) / (200 - 10)

        )

        logits = pfnmixture.forward(
            context_x=target_task_context_x,
            context_y=target_task_context_y,
            query_x=target_task_query_x,
            temperature=1
        )

        nll = pfn_backend.criterion(
            logits,
            target_task_query_y
        ).mean()

    # Plot ---------------
    with (tempfile.TemporaryDirectory() as tmpdirname):
        pfnmixture.logger.reset()  # to check the nll reliability scores

        logger = BufferedFileLogger(
            file_name="distill.csv",
            file_path=tmpdirname,
            buffer_size=1000,
            header=("metric", "value", "context_size", 'global_step'))

        CONTEXT_SIZES = range(10, target_task_context_x.shape[0], 20)

        evaluator = TestOnNewTaskNLL(
            criterion=pfn_backend.criterion,
            logger=logger, device=device
        )

        # PFN Mixture distillation ----------
        evaluator.test_on_new_task(
            model=pfnmixture,
            prefix_x=torch.tensor([]),  # TODO make this default?
            prefix_y=torch.tensor([]),
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            task_name="pfn_mixture",
            step=0,
            context_sizes=CONTEXT_SIZES,
            # fwd kwargs
            temperature=1.0
        )

        # plot the reliability scores for each of the related tasks:
        # TODO for multiple tasks first collect the dataframe then plot
        ax = trainlogger.plot_scalar_curve(
            'reliability_score',
            plot=False,
            x='context_size',
            y='value',
            title='NLL Reliability over context_sizes',
        )
        ax.set_xlabel("Task's Context size")
        ax.set_ylabel("NLL Loss")
        plt.show()

        # No distillation: Should be what the ifbo paper reports

        evaluator.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (no distillation)',
            # prefix_x=baseline_x,
            # prefix_y=baseline_y,
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # adding in half of the related dataset in
        half_x = x_task_context.shape[0] // 2
        evaluator.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (half context task 0)',
            prefix_x=x_task_context[:half_x],
            prefix_y=y_task_context[:half_x],
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # Adding in (almost) the entire context of the related task
        # we can't fit the entire one, since we need space in the sequence
        # to do a batched evaluation over the query points
        evaluator.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (approx. complete context task 0)',
            prefix_x=x_task_context,
            prefix_y=y_task_context,
            context_task_x=target_task_context_x[:-25],
            context_task_y=target_task_context_y[:-25],
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES  # [:10]
        )

        ax = None
        plots = [
            'pfn_mixture',
            'baseline (no distillation)',
            'baseline (approx. complete context task 0)',
            'baseline (half context task 0)'
        ]
        for metric in plots:
            try:
                ax = logger.plot_scalar_curve(
                    metric=metric,
                    plot=False,
                    ax=ax,
                    x='context_size',
                    y='value',
                )
            except ValueError:
                # log.warning(f"Metric {metric} not found in logger.")
                continue

        ax.set_xlabel("Task's Context size")
        ax.set_ylabel("NLL Loss")
        ax.set_title("PFN_Mixture + increments of the new Task (nll)", )
        ax.legend()
        plt.show()
