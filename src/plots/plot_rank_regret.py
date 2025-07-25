import io
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fire

from plots.plot_acq_task_improvement import compute_anytime_performance


def parse_neps_dir(df: pd.DataFrame, pattern, column) -> pd.DataFrame:
    """
    Parse NEPS directory structure from the 'filepath' column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'filepath' column.

    Returns:
        pd.DataFrame: DataFrame with an additional 'neps_dir' column.
    """
    parsed = df[column].apply(lambda x: os.path.basename(os.path.dirname(x))).str.extract(
        pattern)
    # convert to numeric where appropriate
    parsed = parsed.apply(pd.to_numeric, errors='ignore')
    df = df.join(parsed, rsuffix='_parsed')
    return df

def parse_reliab_df(df: pd.DataFrame) -> pd.DataFrame:
    # Collect the target task reliability and entropy of the related reliability scores ------------
    # hacking of indices for compatability
    folds = {
        'taskset': (
            [[16, 8, 6, 17, 4], [2, 5, 18, 9, 7], [19, 3, 0, 21, 15]],
            [[2, 19, 6, 7, 21], [1, 16, 0, 15, 23], [22, 9, 8, 12, 11]],
            [[9, 4, 10, 5, 17], [1, 2, 7, 20, 19], [18, 11, 23, 13, 15]]),
        'lcbench': (
            [[32, 26, 30, 8, 13], [5, 17, 14, 31, 24], [1, 12, 6, 23, 4], [18, 21, 19, 9, 7]],
            [[4, 2, 23, 25, 10], [31, 21, 30, 20, 18], [6, 13, 7, 34, 1], [16, 0, 15, 5, 11]],
            [[25, 24, 10, 4, 6], [3, 27, 5, 23, 20], [34, 21, 29, 17, 2], [7, 33, 31, 18, 11]]
        ),
        'pd1': (
            [[14, 1, 10, 13, 8], [6, 18, 4, 9, 7], [20, 3, 0, 21, 15]],
            [[2, 20, 6, 7, 22], [1, 16, 0, 15, 24], [23, 9, 8, 12, 11]],
            [[22, 4, 10, 5, 19], [1, 2, 7, 21, 20], [18, 11, 24, 13, 15]]
        )
    }

    # Create a mapping from train_id to fold index
    fold_mapping = {}

    for dataset, splits in folds.items():
        fold_mapping[dataset] = {}
        for split_seed, split in enumerate(splits):  # splits is list of folds per split_seed
            fold_mapping[dataset][split_seed] = {}
            for fold_idx, fold_list in enumerate(split):
                # (dataset, split_seed, train_id) → fold_idx
                fold_mapping[dataset][split_seed][str(fold_list)] = fold_idx

    index_tuples = []
    fold_indices = []

    for dataset, split_map in fold_mapping.items():
        for split_seed, train_map in split_map.items():
            for train_id, fold_idx in train_map.items():
                index_tuples.append((dataset, split_seed, train_id))
                fold_indices.append(fold_idx)

    mapping_series = pd.Series(
        fold_indices,
        index=pd.MultiIndex.from_tuples(index_tuples, names=["dataset", "split_seed", "train_id"])
    )

    # Now map using this Series:
    df['train_ids'] = df.set_index(
        ['benchmark.meta.name', 'split_seed', 'train_ids']
    ).index.map(mapping_series).to_list()

    return df

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
        input_file = [input_file]

    input_paths = [Path(f) for f in input_file]

    if {'epochs', 'epoch'} in set(df.columns):  # synthetic data has epochs column :/
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
    df.rename(columns={'split_seed': 'split_seed1'}, inplace=True)
    df = parse_neps_dir(df, pattern, column='filepath')
    df = compute_anytime_performance(df, minimize=True)

    # Calculate normalized regret ------------------------------------
    # Define grouping columns consistently
    group_cols = ['benchmark.meta.name', 'target_task', 'fold', 'split_seed']

    # Create a pivot table: index by group_cols + 'epoch', columns are algorithms, values are anytime_performance
    pivot_df = df.pivot_table(index=group_cols + ['step'], columns='algoname',
                              values='anytime_performance')

    algonames = pivot_df.columns  # Algorithm column names

    # Reset index for easier groupby and merging
    pivot_df_reset = pivot_df.reset_index()

    # Compute best loss per group over all epochs & algorithms (lowest anytime_performance)
    best_group_cols = ['benchmark.meta.name', 'target_task', 'split_seed']
    best_loss_per_group = pivot_df_reset.groupby(best_group_cols).apply(
        lambda group: group.loc[:, algonames].min().min()
    ).rename('best_loss').reset_index()

    # Merge best loss back to the pivot dataframe
    pivot_df_reset = pivot_df_reset.merge(best_loss_per_group, on=best_group_cols)

    # Subtract best_loss from each algorithm's anytime_performance for normalized regret
    pivot_df_reset[algonames] = pivot_df_reset[algonames].subtract(
        pivot_df_reset['best_loss'], axis=0
    )

    # Weight time series of reliability scores ----------------------
    reliab_df = [pd.read_csv(f.parent / 'joint_results.csv') for f in input_paths]
    if len(input_paths) > 1:
        reliab_df = pd.concat(reliab_df, ignore_index=True)
    else:
        reliab_df = reliab_df[0]

    reliab_df = reliab_df[reliab_df['metric'] == 'reliability']
    del reliab_df['metric']  # drop metric column
    reliab_df.rename(columns={'reliability_0': 'target_reliability'}, inplace=True)

    reliab_df = parse_reliab_df(reliab_df)

    # get the highest and second highest weights
    others = reliab_df.filter(like='reliability_')
    # Row-wise max
    row_max_series = others.max(axis=1)

    # Row-wise second max
    # Sort each row descending and take the second value
    row_sorted = others.apply(lambda row: row.sort_values(ascending=False).values, axis=1)
    second_max_series = row_sorted.apply(lambda x: x[1])

    # Add these as new columns
    reliab_df['max_reliability'] = row_max_series
    reliab_df['second_max_reliability'] = second_max_series



    # Plotting --------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns
    import multiprocessing
    import io
    from PIL import Image
    import numpy as np


    # Main code
    benchmarks = pivot_df_reset['benchmark.meta.name'].unique()
    figsize = (5 * len(benchmarks), 4)  # example sizing

    args_list = [(bench, pivot_df_reset, reliab_df, algonames) for bench in benchmarks]

    with multiprocessing.Pool() as pool:
        results = pool.map(plot_single_benchmark, args_list)

    # Create final figure to assemble images
    fig, axes = plt.subplots(1, len(benchmarks), figsize=figsize, sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, (bench, img_bytes) in zip(axes, results):
        # Read image from bytes
        image = Image.open(io.BytesIO(img_bytes))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Benchmark: {bench}')

    # Optional: set only the first y-label for normalized regret
    axes[0].set_ylabel('Normalized Regret')

    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def plot_single_benchmark(args):
    bench, pivot_df_reset, reliab_df, algonames = args

    # Filter data for this benchmark
    bench_data = pivot_df_reset[pivot_df_reset['benchmark.meta.name'] == bench]
    rel_data = reliab_df[reliab_df['benchmark.meta.name'] == bench]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Primary axis: normalized regret lines for each alg
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f'Benchmark: {bench}')
    ax.set_xlabel('Step')
    for alg in algonames:
        sns.lineplot(data=bench_data, x='step', y=alg, label=alg, ax=ax)
    ax.set_ylabel('Normalized Regret')

    # Secondary y-axis
    ax2 = ax.twinx()
    sns.lineplot(data=rel_data, x='step', y='target_reliability', color='red',
                 label='Target Reliability', ax=ax2)
    sns.lineplot(data=reliab_df, x='step', y='max_reliability',
                 label='highest "other" weight', ax=ax2)
    sns.lineplot(data=reliab_df, x='step', y='second_max_reliability',
                 label='second highest "other" weight', ax=ax2)

    ax2.set_ylabel('Target Reliability')
    ax2.grid(False)

    # Save to bytes buffer in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return bench, buf.read()

if __name__ == '__main__':
    fire.Fire(main)

    # synthetic only:
    #  --input_file "['/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/synthetic-final/anytime.csv']"

    # real benchmarks:
    # --input_file "['/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/lcbench-folds-split_seed/anytime.csv','/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/pd1-folds-split_seed/anytime.csv','/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/07-24/taskset-folds-split_seed/anytime.csv']"
