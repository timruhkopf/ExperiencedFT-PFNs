from typing import Dict, Any

from ifbo import Batch


def parse_batch(batch:Batch, target_idx:int, n_prefix_tokens:int=0) -> Dict[str, Any]:
    """
    Parses a batch, extracting context/query splits for the target task (at target_idx)
    and the context for all related tasks (excluding the target).
    :param batch: the batch to parse
    :param target_idx: the index of the target task in the batch
    :param n_prefix_tokens: number of prefix tokens to use for the target task.
    eventually will limit the number of query points to 1000 - target_task_context - n_prefix_tokens

    :returns Dict with
    - target_task_context: dict with x and y for the context of the target task
    - target_task_query: dict with x and y for the query of the target task
    - related_task_data: list of dicts with x and y for the context of the related tasks
    """
    sep = batch.single_eval_pos[0]
    x, y = batch.x, batch.y

    # 0 task is the one we want to predict on
    target_task_context_x = x[:sep, target_idx:target_idx + 1]
    target_task_context_y = y[:sep, target_idx:target_idx + 1]
    target_task_val_x = x[sep:, target_idx:target_idx + 1]
    target_task_val_y = y[sep:, target_idx:target_idx + 1]

    # fixme: max context size is 1k, so distilled + task context + query can
    #  exceed that. in that case we will want to batch over the exceeing query points with the
    #  same context.
    max_query_size = 1000 - target_task_context_x.shape[0] - n_prefix_tokens
    target_task_query_x = x[sep:sep + max_query_size, 0:1]  # actual test points
    target_task_query_y = y[sep:sep + max_query_size, 0:1]  # actual test labels


    # x_val = x[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    # y_val = y[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    # print(x.shape, y.shape)

    related_task_indices = [i for i in range(x.shape[1]) if i != target_idx]
    related_task_data = [
        {
            'x': x[:sep, i:i + 1],
            'y': y[:sep, i:i + 1],
        }
        for i in related_task_indices
    ]

    return {
        'target_task_context': {
            'x': target_task_context_x,
            'y': target_task_context_y
        },

        'target_task_query': {
            'x': target_task_query_x,
            'y': target_task_query_y
        },

        'related_task_data': related_task_data
    }
