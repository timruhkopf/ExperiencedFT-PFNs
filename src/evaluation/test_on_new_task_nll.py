import warnings

import torch


class TestOnNewTaskNLL:
    def __init__(self, criterion, logger, device):
        self.criterion = criterion
        self.logger = logger
        self.device = device

    @torch.no_grad()
    def test_on_new_task(
            self,
            model,
            task_name,
            context_task_x,
            context_task_y,
            query_task_x,
            query_task_y,
            prefix_x=torch.tensor([]),
            prefix_y=torch.tensor([]),
            context_sizes=None,
            step=None,
            **kwargs
    ):
        """
        Evaluate the distilled context on a new task with nll loss over the contextsizes of the
        target task.

        i.e. given the (prefix_x, context_y) distillation from a related task,
        we will evaluate the performance on the different context_sizes on the new task
        (context_task_x, context_task_y) and evaluate the nll loss on the holdout
        (query_task_x, query_task_y) on that task.
        :param prefix_x: distilled context points (T x B x dim),
        :param prefix_y: distilled context labels (T x B)
        :param context_task_x: task context points (T x B x dim), will be iterated over acc. to context_sizes
        :param context_task_y: task context labels (T x B), will be iterated over acc. to context_sizes
        :param query_task_x: task query points (T x B x dim), (hold-out)
        :param query_task_y: task query labels (T x B), (hold-out)
        :param context_sizes: list of context sizes to evaluate
        :param task_name: name of the task for logging
        :param step: "global step" for logging; i.e. if evaluated during training.
        :param kwargs: additional arguments for the model's forward

        :return: list of losses for each context size
        """

        if context_sizes is None:
            context_sizes = [context_task_x.shape[0]]

        assert prefix_x.shape[0] + context_task_x.shape[0] <= 1000, \
            f"Distilled context + task context exceeds pfn's 1000 tokens. " \
            f"Distilled context: {prefix_x.shape[0]}, task context: {context_task_x.shape[0]}" \
            f"Cannot predict for query_task_x: {query_task_x.shape[0]}"

        if prefix_x.shape[0] + context_task_x.shape[0] + query_task_x.shape[0] >= 1000:
            warnings.warn(

                f"Distilled context + task context + query exceeds pfn's 1000 tokens. " \
                f"Distilled context: {prefix_x.shape[0]}, task context: {context_task_x.shape[0]}, " \
                f"query: {query_task_x.shape[0]} --> Attempting batch prediction over query points."
            )

        prefix_x, prefix_y = prefix_x.to(self.device), prefix_y.to(self.device)
        context_task_x, context_task_y = context_task_x.to(self.device), context_task_y.to(
            self.device)

        losses = []
        for context_size in context_sizes:  # note how the final y is omitted here
            # batching of query points in case we exceed the context size
            query_size = 1000 - prefix_x.shape[0] - context_size
            batch_losses = []
            for i in range(0, query_task_x.shape[0], query_size):
                logits = model(
                    (  # distilled context + observed x part of task, query for that task
                        torch.cat([prefix_x, context_task_x[:context_size],
                                   query_task_x[i:i + query_size]], dim=0),
                        # distilled labels + observed y part of task,
                        torch.cat([prefix_y, context_task_y[:context_size]], dim=0)
                    ),
                    single_eval_pos=prefix_x.shape[0] + \
                                    min(context_size, context_task_x.shape[0]),

                    **kwargs
                )
                # y's associated with query for that task
                target = query_task_y[i:i + query_size]
                loss = self.criterion(logits, target)

                loss = loss.view(-1, logits.shape[1])  # sometimes the seq length can be one off
                batch_losses.append(loss)

            loss = torch.mean(torch.cat(batch_losses))
            losses.append(loss.item())

            self.logger.add_scalar(
                task_name,
                loss.item(),
                context_size, step
            )

        return losses
