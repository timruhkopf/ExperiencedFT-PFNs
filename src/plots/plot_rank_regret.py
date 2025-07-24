import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fire

from plots.plot_acq_task_improvement import compute_anytime_performance


def parse_neps_dir(df: pd.DataFrame, pattern) -> pd.DataFrame:
    """
    Parse NEPS directory structure from the 'filepath' column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'filepath' column.

    Returns:
        pd.DataFrame: DataFrame with an additional 'neps_dir' column.
    """
    parsed = df['filepath'].apply(lambda x: os.path.basename(os.path.dirname(x))).str.extract(
        pattern)
    # convert to numeric where appropriate
    parsed = parsed.apply(pd.to_numeric, errors='ignore')
    df.rename(columns={'split_seed': 'split_seed1'}, inplace=True)
    df = df.join(parsed)
    return df

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(input_file, output_file=None, figsize=(15, 6),
         pattern=(
         r'neps_root_directory_(?P<target_task>[^_]+)_(?P<fold>[^_]+)_(?P<split_seed>[^_]+)_(?P<seed>[^_]+)')):
    """
    Args:
        input_file (str or list): Path(s) to the input CSV file(s).
        output_file (str, optional): Path to save the plot. If None, display the plot.
        figsize (tuple): Size of the figure.
        pattern (str): Regex pattern to extract metadata from directory path.
    """

    # Load data
    if isinstance(input_file, list):
        df = pd.concat([pd.read_csv(f) for f in input_file],
                       ignore_index=True)

    else:
        df = pd.read_csv(input_file)

    if {'epochs', 'epoch'} in set(df.columns) : # synthetic data has epochs column :/
        # where it is not Nan, copy the value to the epoch column
        df['epoch'] = df['epochs'].where(df['epochs'].notna(), df['epoch'])
        del df['epochs']
    elif 'epochs' in df.columns:
        # if epochs is present but empty, drop it
        df['epoch'] = df['epochs']
        del df['epochs']
    # Assume these two functions are defined somewhere else:
    # parse_neps_dir: parses metadata from path columns based on pattern
    # compute_anytime_performance: computes 'anytime_performance' column
    df = parse_neps_dir(df, pattern)

    df = compute_anytime_performance(df, minimize=True)


    # Define grouping columns consistently
    group_cols = ['benchmark.meta.name', 'target_task', 'fold', 'split_seed']

    # Create a pivot table: index by group_cols + 'epoch', columns are algorithms, values are anytime_performance
    pivot_df = df.pivot_table(index=group_cols + ['step'], columns='algoname',
                              values='anytime_performance')

    algonames = pivot_df.columns  # Algorithm column names

    # Reset index for easier groupby and merging
    pivot_df_reset = pivot_df.reset_index()

    # Compute best loss per group over all epochs & algorithms (lowest anytime_performance)
    best_group_cols =  ['benchmark.meta.name', 'target_task', 'split_seed']
    best_loss_per_group = pivot_df_reset.groupby(best_group_cols).apply(
        lambda group: group.loc[:, algonames].min().min()
    ).rename('best_loss').reset_index()

    # Merge best loss back to the pivot dataframe
    pivot_df_reset = pivot_df_reset.merge(best_loss_per_group, on=best_group_cols)

    # Subtract best_loss from each algorithm's anytime_performance for normalized regret
    pivot_df_reset[algonames] = pivot_df_reset[algonames].subtract(pivot_df_reset['best_loss'], axis=0)

    # Plotting
    benchmarks = pivot_df_reset['benchmark.meta.name'].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(benchmarks), figsize=figsize, sharey=True)

    # If only one benchmark, axes might not be iterable, fix that:
    if len(benchmarks) == 1:
        axes = [axes]

    axes[0].set_ylabel('Normalized Regret')

    for ax, bench in zip(axes, benchmarks):
        bench_data = pivot_df_reset[pivot_df_reset['benchmark.meta.name'] == bench]

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Zero line reference
        ax.set_title(f'Benchmark: {bench}')
        ax.set_xlabel('Step')


        for alg in algonames:
            sns.lineplot(data=bench_data, x='step', y=alg, label=alg, ax=ax)


    axes[-1].legend(title='Algorithm')

    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


if __name__ == '__main__':
    fire.Fire(main)

    # synthetic only:
    #  --input_file "['/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/synthetic-final/anytime.csv']"

    # real benchmarks:
    # --input_file "['/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/lcbench-folds-split_seed/anytime.csv','/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/pd1-folds-split_seed/anytime.csv','/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/taskset-folds-split_seed/anytime.csv']"
