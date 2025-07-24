from itertools import chain
from typing import Dict, List

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
        **kwargs
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


def linear_alg(num_tasks, context_y, imputed_y, device, lambda_reg=0.1):
    # context_y: (N,)
    # imputed_y: (T, N)
    imputed_y = imputed_y  # Ensure shape is (N, T) for downstream code
    # Goal: solve y = α * x + β  for each task
    y = context_y.view(-1).to(device)            # (N,)
    x = imputed_y.to(device)                     # (N, T)
    x_mean = x.mean(dim=0, keepdim=True)         # (1, T)
    y_mean = y.mean()                            # scalar
    x_centered = x - x_mean                      # (N, T)
    y_centered = y.unsqueeze(1) - y_mean         # (N, 1)
    # Compute covariance between each x[:,j] and y
    cov_xy = (x_centered * y_centered).mean(dim=0)  # (T,)
    var_x = (x_centered ** 2).mean(dim=0)           # (T,)
    # Regularized alpha
    alpha = (cov_xy + lambda_reg) / (var_x + lambda_reg)  # (T,)
    # Compute beta for each task
    beta = y_mean - alpha * x_mean.squeeze(0)            # (T,)
    # Project y back into x space: x_proj = (y - beta) / alpha
    y_proj = (y.unsqueeze(1) - beta.unsqueeze(0)) / alpha.unsqueeze(0)  # (N, T)
    return y_proj

def linear_alg_main(num_tasks, context_x, context_y, imputed_y, device):
    x = torch.arange(context_x.shape[0], device=context_x.device).reshape(-1, 1)  # [num_points, 1]
    degree = 1
    x = torch.cat([x ** i for i in range(degree+1)], dim=1).to(device)  # Polynomial features
    X_design = torch.cat([context_y.unsqueeze(1), x], dim=1).to(device)  # Add context_y as first

    # Build block-diagonal design matrix for all tasks
    X_design_block = torch.block_diag(*[X_design for _ in range(num_tasks)])  # [num_tasks*num_points, ...][2][5]
    # Reorder imputed_y to match block-diagonal structure: all points for task 0, then task 1, etc.
    imputed_y_ordered = imputed_y.transpose(0, 1).contiguous().view(-1)  # [num_tasks*num_points]

    beta = torch.linalg.lstsq(X_design_block, imputed_y_ordered).solution
    y_proj = X_design_block @ beta
    y_proj = y_proj.clamp(0, 1)
    
    return y_proj   


def norm_alg(num_tasks, context_y, imputed_y, device):
    """
    Normalize imputed_y for each task to [0, 1], then scale to context_y's range.
    Returns y_proj of shape [num_points, num_tasks].
    """
    context_y = context_y.to(device).unsqueeze(1)  # [num_points, 1]
    imputed_y = imputed_y.to(device)  # [num_tasks, num_points]

    imputed_y_min = imputed_y.min(dim=0, keepdim=True).values  # [num_tasks, 1]
    imputed_y_max = imputed_y.max(dim=0, keepdim=True).values  # [num_tasks, 1]
    # Avoid division by zero
    denom = (imputed_y_max - imputed_y_min).clamp(min=1e-3)
    n_imputed_y = (imputed_y - imputed_y_min) / denom  # [num_tasks, num_points ]
    context_y_min = context_y.min()
    context_y_max = context_y.max()
    y_proj = n_imputed_y * (context_y_max - context_y_min) + context_y_min  # [num_tasks, num_points]
    return y_proj


def calc_imputed_linalg_reliability(
        model: TransformerModel,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        related_task_data: Dict[str, MyBatch],  # type: ignore
        criterion: BarDistribution,
        verbose: bool = False,
        plot_file_path: str = None,  # type: ignore
        degree_fn=lambda x, y :0,
        multi_fidelity = True,
        transformation_type=None
) -> torch.Tensor:
    """
    Compute scale-invariant reliability scores for meta-tasks using block-diagonal regression.

    Performs per-task imputation followed by independent linear regression in a block-diagonal feature space
    to estimate task reliability while being invariant to linear scaling of learning curves.

        :param model: Transformer meta-model used for cross-task imputation
        :type model: TransformerModel
        :param context_x: Target task's context features (fidelity + hyperparameters),
                          shape [num_points, num_features]
        :type context_x: torch.Tensor
        :param context_y: Target task's observed outputs (e.g., validation accuracy),
                          shape [num_points, 1]
        :type context_y: torch.Tensor
        :param related_task_data: Dictionary of batched meta-task data containing:
            - x: Context features for each meta-task, shape [num_meta_tasks, num_points, num_features]
            - y: Context outputs for each meta-task, shape [num_meta_tasks, num_points, 1]
            - padding_mask: Boolean mask for variable-length contexts
        :type related_task_data: Dict[str, MyBatch]
        :param criterion: Distribution object providing:
            - median(): For deterministic imputation
            - __call__(): For NLL computation between logits and targets
        :type criterion: BarDistribution
        :param verbose: If True, save diagnostic plot to specified path.
        :param plot_file_path: Output path for diagnostic plot when verbose=True.

    :return: Reliability scores (mean NLL) per meta-task, shape [num_meta_tasks]
    :rtype: torch.Tensor

    :Notes:
        - Robust to affine transformations: The regression step makes reliability scores invariant to linear scaling/shifting of learning curves
        - Block-diagonal design: Uses kronecker product to create independent design matrices while maintaining computational efficiency
        - Debug plotting: Set internal flag to visualize imputed vs projected values per task (uses Matplotlib)
        - Time complexity: O((num_meta_tasks * num_points)^3) due to block-diagonal least squares
    """
    device = context_x.device
    context_x = context_x.unsqueeze(1)

    # for task_data in related_task_data:
    task_context_x = related_task_data.x
    task_context_y = related_task_data.y
    padding_mask = related_task_data.padding_mask
    num_related = task_context_x.shape[1]

    # impute target_task y's for conditioned on each related task --------------
    #print("calc_imputed_linalg_reliability")
    #print(task_context_y.shape)
    #print(task_context_y[:,:4])
    logits = model(
        (
            torch.cat([task_context_x, context_x.repeat(1, num_related, 1)], dim=0),
            task_context_y
        ),
        single_eval_pos=task_context_x.shape[0],
        src_key_padding_mask=padding_mask
    )
    imputed_y = criterion.median(logits)  # shape [num_points, num_tasks]

    # Learn the projection from the related task to the target task ------------
    if multi_fidelity:
        target_fidelity = context_x[:, 0, 1]
        x = target_fidelity.reshape(-1, 1)  # Ensure x is column vector
        degree = degree_fn(context_x, context_y)
        
        x = torch.cat([x ** i for i in range(degree+1)], dim=1).to(device)  # Polynomial features
        X_design = torch.cat([context_y.unsqueeze(1), x], dim=1).to(device)  # Add context_y as first
        # column

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
        if verbose:
            plot_projections(
                target_fidelity=target_fidelity,
                context_x=context_x,
                context_y=context_y,
                imputed_y=imputed_y,
                y_proj=y_proj,
                plot_file_path=plot_file_path
            )

        num_fidelity = target_fidelity.shape[0]
        num_tasks = imputed_y.shape[1]

        # compute the reliability scores (nll) based on the projected y ------------
        # y's associated with query for that task
        # target = context_y.repeat(1, num_related)
        # Reshape y_proj to [num_points, num_tasks] for loss computation
        y_proj_for_loss = y_proj.view(num_tasks, num_fidelity).T  # [num_points, num_tasks]
        loss = criterion(logits, y_proj_for_loss)
        loss = loss.view(-1, logits.shape[1])  # bar distribution issue
        loss = loss.mean(dim=0)  # mean over the batch

        return loss  # reliability scores

    else:
        num_tasks = imputed_y.shape[1]
        num_fidelity = imputed_y.shape[0]

        if transformation_type == "cosine":
            num_tasks = imputed_y.shape[1]
            num_fidelity = imputed_y.shape[0]
            z_mean_imputed_y = imputed_y - imputed_y.mean(dim=0)
            z_mean_context_y = context_y - context_y.mean(dim=0)
            cosine_similarity = torch.nn.functional.cosine_similarity(z_mean_context_y, z_mean_imputed_y.T)
            #print(f"Cosine similarity: {cosine_similarity}")
            return 1 - cosine_similarity

        else:
            # compute the reliability scores (nll) based on the projected y ------------
            # y's associated with query for that task
            # target = context_y.repeat(1, num_related)
            # Reshape y_proj to [num_points, num_tasks] for loss computation
            num_tasks = imputed_y.shape[1]
            if transformation_type == "linear":
                y_proj = linear_alg(num_tasks, context_y, imputed_y, device).T
            elif transformation_type == "norm":
                y_proj = norm_alg(num_tasks, context_y, imputed_y, device).T
            else:
                y_proj = linear_alg_main(num_tasks, context_x, context_y, imputed_y, device).T
            #y_proj = norm_alg(num_tasks, context_y, imputed_y, device).T
            #y_proj = linear_alg_main(num_tasks, context_x, context_y, imputed_y, device).T
            y_proj_for_loss = y_proj.view(num_tasks, num_fidelity).T  # [num_points, num_tasks]
            loss = criterion(logits, y_proj_for_loss)
            loss = loss.mean(dim=0)  # mean over the batch
            return loss         

def plot_projections(
        target_fidelity: torch.Tensor,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        imputed_y: torch.Tensor,
        y_proj: torch.Tensor,
        plot_file_path: str
):
    import numpy as np
    import matplotlib.pyplot as plt

    target_y = context_y
    num_fidelity = target_fidelity.shape[0]
    num_tasks = imputed_y.shape[1]

    # Reshape y_proj for per-task plotting (now correct shape)
    y_proj_reshaped = y_proj.cpu().numpy().reshape(num_tasks,
                                                   num_fidelity).T  # shape [num_fidelity, num_tasks]
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
    if plot_file_path:
        plt.savefig(plot_file_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import KFold  # or GroupKFold for stratification
from torch.nn.utils.rnn import pad_sequence


def build_padded_batch(context_x, context_y, indices_grouped):
    # Each group is a list of indices for one curve/HP config
    x_seqs = [context_x[idxs] for idxs in indices_grouped]  # [curve_len, feat_dim]
    y_seqs = [context_y[idxs] for idxs in indices_grouped]  # [curve_len, ...]
    padded_x = pad_sequence(x_seqs, batch_first=True)  # [batch, max_len, feat_dim]
    padded_y = pad_sequence(y_seqs, batch_first=True)  # [batch, max_len, ...]
    lengths = torch.tensor([len(seq) for seq in x_seqs])
    max_len = padded_x.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return padded_x, padded_y, mask


def kfold_hp_split(context_x, context_y, n_splits=5, random_state=42, start_feature_indx = 2):
    """
    Splits data so that all tokens from a given HP config are held out together.
    Returns context (train) and query (test) sets for the specified fold.
    """
    if context_x.ndim == 3:
        cx = context_x.squeeze(1).cpu().numpy() 
    else:
        cx = context_x.cpu().numpy()  # shape: [n_tokens, n_features]

    n_tokens = cx.shape[0]
    fidelity_col = 1
    hp_cols = list(range(start_feature_indx, cx.shape[1])) 

    # Group indices by HP configuration
    curve_indices = defaultdict(list)
    for i in range(n_tokens):
        hp_tuple = tuple(cx[i, hp_cols].tolist())
        curve_indices[hp_tuple].append(i)

    # List of unique HP configs and their associated token indices
    hp_tuples = list(curve_indices.keys())
    hp_indices = list(curve_indices.values())

    # KFold split on HP configs
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    train_groups = []
    test_groups = []
    for train_hp_idx, test_hp_idx in kf.split(hp_tuples):
        # Flatten token indices for train/test HPs
        train_indices = [idx for i in train_hp_idx for idx in hp_indices[i]]
        test_indices = [idx for i in test_hp_idx for idx in hp_indices[i]]
        splits.append((np.array(train_indices), np.array(test_indices)))

        # collect the train_groups and test_groups; i.e. collect the learning curve tokens
        # associated with each HP config

        train_groups.append(list(chain(*[hp_indices[i] for i in train_hp_idx])))
        test_groups.append(list(chain(*[hp_indices[i] for i in test_hp_idx])))

    # Pad and batch
    padded_context_x, padded_context_y, context_mask = build_padded_batch(
        context_x,
        context_y,
        train_groups
    )
    padded_query_x, padded_query_y, query_mask = build_padded_batch(
        context_x,
        context_y,
        test_groups
    )

    return padded_context_x, padded_context_y, ~context_mask, \
        padded_query_x, padded_query_y, ~query_mask

def calc_target_cv_nll(context_x, context_y, model, criterion, splits=5, random_state=42, start_feature_indx=2):

    device = context_x.device

    padded_context_x, padded_context_y, context_mask, \
        padded_query_x, padded_query_y, query_mask = kfold_hp_split(
        context_x, context_y, n_splits=splits, random_state=random_state, start_feature_indx=start_feature_indx
    )

    # Concatenate context and query for model input
    all_x = torch.cat([padded_context_x, padded_query_x], dim=1)
    all_x = all_x.permute(1, 0, 2).to(device) # [batch_size, seq_len, feature_dim]

    padded_context_y = padded_context_y.permute(1, 0).to(device)
    padded_query_y = padded_query_y.permute(1, 0).to(device)

    kf_logits = model((all_x, padded_context_y),
                      single_eval_pos=padded_context_x.shape[1],
                      src_key_padding_mask=context_mask.to(device))

    # Compute loss on the (unpadded) query set
    kf_loss = criterion(kf_logits, padded_query_y)
    kf_loss = kf_loss[~query_mask.T.to(device)].mean(dim=0)

    return kf_loss