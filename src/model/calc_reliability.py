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


def calc_imputed_linalg_reliability(
        model: TransformerModel,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        related_task_data: Dict[str, MyBatch],  # type: ignore
        criterion: BarDistribution,
) -> torch.Tensor:
    """
    Calculate the reliability of the related tasks with respect to the current task using imputed linalg.

    Args:
        :param model: The model to use for the calculation.
        :param context_x: The context points of the current task.
        :param context_y: The context values of the current task.
        :param related_task_data: The data of the related tasks.
        :param criterion: The criterion to use for the calculation.

    Returns:
        A tensor containing the reliability scores for each related task.
    """
    device = context_x.device

    # for task_data in related_task_data:
    task_context_x = related_task_data.x
    task_context_y = related_task_data.y
    padding_mask = related_task_data.padding_mask
    num_related = task_context_x.shape[1]

    logits = model(
        (
            torch.cat([task_context_x, context_x.repeat(1, num_related, 1)], dim=0),
            task_context_y
        ),
        single_eval_pos=task_context_x.shape[0],
        src_key_padding_mask=padding_mask
    )

    imputed_y = criterion.median(logits)  # shape [num_points, num_tasks]

    target_fidelity = context_x[:, 0, 1]
    # Build polynomial features for target fidelity & then the design matrix
    x = target_fidelity.reshape(-1, 1)  # Ensure x is column vector
    degree = 0
    x = torch.cat([x ** i for i in range(degree + 1)], dim=1).to(device)  # Polynomial features
    X_design = torch.cat([context_y, x], dim=1).to(device)  # Add context_y as first column

    # Build block-diagonal design matrix for all tasks
    X_design_block = torch.block_diag(*[X_design for _ in range(num_related)])  # [num_tasks*num_points, ...][2][5]

    # Reorder imputed_y to match block-diagonal structure: all points for task 0, then task 1, etc.
    imputed_y_ordered = imputed_y.transpose(0, 1).contiguous().view(-1)  # [num_tasks*num_points]

    beta = torch.linalg.lstsq(X_design_block, imputed_y_ordered).solution
    y_proj = X_design_block @ beta
    y_proj = y_proj.clamp(0, 1)

    num_tasks = imputed_y.shape[1]
    num_fidelity = target_fidelity.shape[0]

    # For plotting and debugging
    if False:
        import numpy as np
        import matplotlib.pyplot as plt

        target_y = context_y
        num_fidelity = target_fidelity.shape[0]
        num_tasks = imputed_y.shape[1]

        # Reshape y_proj for per-task plotting (now correct shape)
        y_proj_reshaped = y_proj.cpu().numpy().reshape(num_tasks, num_fidelity).T  # shape [num_fidelity, num_tasks]
        imputed_np = imputed_y.cpu().numpy()
        fidelity = target_fidelity.cpu().numpy()
        target_y_np = target_y.cpu().numpy()  # ground truth for target task

        fig, axs = plt.subplots(1, num_tasks + 1, figsize=(4 * (num_tasks + 1), 5), sharey=True)
        fig.suptitle('Per-Task Projection Analysis', fontsize=16)

        # Plot ground truth (target task)
        axs[0].scatter(fidelity, target_y_np, label='Ground Truth')
        axs[0].set_title('Target Task (Ground Truth)')
        axs[0].set_xlabel('Fidelity')
        axs[0].set_ylabel('y value')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Plot each related/meta task
        for task_idx in range(num_tasks):
            ax = axs[task_idx + 1]
            y_before = imputed_np[:, task_idx]
            y_after = y_proj_reshaped[:, task_idx]
            x = fidelity

            # Imputed (before)
            ax.scatter(x, y_before, color='red', label='Imputed (before)', zorder=3)
            # Projected (after)
            ax.scatter(x, y_after, color='blue', marker='s', label='Projected (after)', zorder=3)
            # Error bars
            for xi, yb, ya in zip(x, y_before, y_after):
                ax.plot([xi, xi], [yb, ya], color='gray', linestyle=':', zorder=2)
            ax.set_title(f'Related Task {task_idx + 1}')
            ax.set_xlabel('Fidelity')
            ax.grid(True, alpha=0.3)
            if task_idx == 0:
                ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # y's associated with query for that task
    # target = context_y.repeat(1, num_related)
    # Reshape y_proj to [num_points, num_tasks] for loss computation
    y_proj_for_loss = y_proj.view(num_tasks, num_fidelity).T  # [num_points, num_tasks]
    loss = criterion(logits, y_proj_for_loss)
    loss = loss.view(-1, logits.shape[1])  # bar distribution issue
    loss = loss.mean(dim=0)  # mean over the batch

    return loss  # reliability scores





