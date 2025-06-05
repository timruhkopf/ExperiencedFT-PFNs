from typing import Dict

import numpy as np
import torch
from ifbo.transformer import TransformerModel
from ifbo import BarDistribution, FTPFN
from src.model.batch_padded_pfn import MyBatch


def _calc_reliability(
        model: TransformerModel,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        related_task_data: MyBatch,  # type: ignore
        criterion: BarDistribution,
        peeking: Dict[str, torch.Tensor] = None
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

    # for task_data in related_task_data:
    task_context_x = related_task_data.x
    task_context_y = related_task_data.y
    padding_mask = related_task_data.padding_mask
    num_related = task_context_x.shape[1]

    if peeking is not None:
        # The ida behind
        x, y = peeking['x'], peeking['y']
        # bootstrap sample without replacement that is at most half the size of the peeking tensor
        # bootstrapping has the disadvantage, that we can immediately predict based of the
        # adjacent fidelities and get poor generalization
        # ids = np.random.choice(
        #     x.shape[0],
        #     size=context_x.shape[0] // 2,
        #     replace=False
        # )
        # FT context is somewhat ordered, so the future fidelities are likely to be
        # later in the context sequence. This gives some of the new tasks context, but
        # the learning task is still important
        ids = torch.arange(x.shape[0] // 2).to(x.device)
        x = x[ids].repeat(1, num_related, 1)
        y = y[ids].repeat(1, num_related)
        # reduce the query (context_x) by those ids
        remaining_ids = np.setdiff1d(
            np.arange(context_x.shape[0]),
            ids
        )
        context_x = context_x[remaining_ids]
        context_y = context_y[remaining_ids]

        task_context_x = torch.cat([x, task_context_x], dim=0)
        task_context_y = torch.cat([y, task_context_y], dim=0)

        # prepend the padding mask
        padding_mask = torch.cat(
            [torch.zeros(num_related, x.shape[0], dtype=torch.bool).to(task_context_x.device),
             padding_mask],
            dim=1
        )

    logits = model(
        (
            torch.cat([task_context_x, context_x.repeat(1, num_related, 1)], dim=0),
            task_context_y
        ),
        single_eval_pos=task_context_x.shape[0],
        src_key_padding_mask=padding_mask
    )
    # y's associated with query for that task
    target = context_y.repeat(1, num_related)
    loss = criterion(logits, target)
    loss = loss.view(-1, logits.shape[1])  # bar distribution issue
    loss = loss.mean(dim=0)  # mean over the batch

    return loss  # reliability scores
