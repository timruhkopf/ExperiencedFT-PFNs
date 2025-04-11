from copy import copy

import ifbo
import torch
from ifbo import Batch
from ifbo.utils import detokenize

from src.dataset.taskprior import MetaTaskPriorSameProblem


def generate_round_robin_batches(batch, max_steps: int, min_context=10, stepsize=10):
    """
    Generates modified batches with incrementing single_eval_pos for target task
    while maintaining original splits for related tasks.

    Args:
        batch: Contains x, y, single_eval_pos as described
        max_steps: Maximum number of evaluation position increments

    Returns:
        Generator yielding (target_idx, modified_batch) tuples
    """
    x, y, original_sep = batch.x, batch.y, batch.single_eval_pos
    batch_size = batch.x.size(1)

    for target_idx in range(batch_size):

        # Determine maximum possible offset for this sequence
        seq_len = x.size(0)
        original_split = original_sep[target_idx]

        for modified_sep in range(min_context, original_split + 1, stepsize):
            # Create new target_batch with modified split positions
            target_batch = Batch(
                x=x[:, target_idx:target_idx + 1],
                y=y[:, target_idx:target_idx + 1],
                target_y=y[:, target_idx:target_idx + 1],
                single_eval_pos=modified_sep
            )

            target_context, target_query = detokenize(target_batch, context_size=modified_sep)

            yield target_idx, target_context, target_query, modified_sep

if __name__ == '__main__':
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=5, single_eval_pos=500)

    # (1) instantiate pfn model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ifbo.surrogate.FTPFN(version="0.0.1", device=device)

    x, y, original_sep = batch.x, batch.y, batch.single_eval_pos
    batch_size = batch.x.size(1)
    original_sep = copy(original_sep)
    results = {}  # Dict[Tuple[target_idx, related, single_eval_pos], float]
    for target_idx, target_context, _, sep in generate_round_robin_batches(batch, max_steps=5):

        assert sep == sum(len(curve.t) for curve in target_context), \
            f"Expected sep {sep}, but got {sum(len(curve.t) for curve in target_context)}"
        context_tasks = [context[i] for i in range(batch_size) if i != target_idx]

        for r, related_task in enumerate(context_tasks):
            print(target_idx, r, sep)
            # Your existing processing pipeline
            # fixme: using the predict method on each combination is super inefficient.
            #  quicker way: collect the context and query tokens and use the fwd directly.
            #  but this will require us to change the way that we need to calcualte the nll as well
            predictions = model.predict(related_task, target_context)

            nll_score = torch.cat([(-prediction.likelihood(q.y).cpu())
                                   for prediction, q in zip(predictions, target_context)])

            results[(target_idx, r, sep)] = nll_score

    # for key, value in results.items():
    #     results[key] = value.item()

    processed_results = [(*k, v.quantile(q=0.5).item(), v.quantile(q=0.95).item(), v.quantile(
        q=0.05).item()) for k, v in results.items()]

    # get the pd.DataFrame from the results
    import pandas as pd

    results_df = pd.DataFrame(processed_results,
                              columns=['Task', 'Related', 'SingleEvalPos', 'median NLL',
                                       'upper NLL', 'lower NLL']
                              )
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Ensure proper sorting for line connections
    results_df = results_df.sort_values(['Task', 'Related', 'SingleEvalPos'])

    # Create FacetGrid with error bands
    g = sns.FacetGrid(results_df, col='Task', col_wrap=3, height=4,
                      sharey=False, despine=False)
    g.map_dataframe(sns.lineplot, x='SingleEvalPos', y='median NLL',
                    hue='Related', style='Related',
                    palette='viridis', markers=True, dashes=False,
                    errorbar=None)  # We'll add CI manually

    # Add error bands using fill_between
    for ax, task in zip(g.axes.flat, results_df['Task'].unique()):
        task_data = results_df[results_df['Task'] == task]
        for rel, color in zip(task_data['Related'].unique(), sns.color_palette('viridis')):
            rel_data = task_data[task_data['Related'] == rel]
            ax.fill_between(rel_data['SingleEvalPos'],
                            rel_data['lower NLL'],
                            rel_data['upper NLL'],
                            color=color, alpha=0.2)

    g.add_legend(title='Related')
    g.set_axis_labels('SingleEvalPos', 'Median NLL with CI')
    g.set_titles("Task {col_name}")
    plt.tight_layout()
    plt.show()