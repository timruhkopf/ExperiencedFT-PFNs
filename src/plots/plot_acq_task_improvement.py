import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def compute_anytime_performance(df):
    # Sort by filepath and epoch to ensure correct order
    df = df.sort_values(['filepath', 'epoch'])
    # Compute anytime performance (cumulative min loss) for each run
    df['anytime_performance'] = df.groupby('filepath')['loss'].cummin()
    # Add arange ids to the dataframe
    df['step'] = df.groupby('filepath').cumcount()
    return df


def compute_per_task_improvement(df_anytime, baseline_algo='ifbox'):
    # Identify the baseline (ifbox) performance per task and step
    baseline_df = df_anytime[df_anytime['algorithm.surrogate_model.meta.name'] == baseline_algo]
    baseline_df = baseline_df[
        ['train_ids', 'target_task',
         # 'split_seed',
         'step', 'anytime_performance']
    ].rename(columns={'anytime_performance': 'ifbox_anytime_performance'})

    # repetitions over the baseline; so we want the median over ifbox
    baseline_df = (
        baseline_df.groupby(['train_ids', 'target_task', 'step'])       # 'split_seed',
        .agg(ifbox_anytime_performance=('ifbox_anytime_performance', 'median'))
    ).reset_index()


    # Merge baseline performance with all runs
    merged = df_anytime.merge(
        baseline_df,
        on=['train_ids', 'target_task',
#             'split_seed',
            'step'],
        how='left'
    )

    merged = merged[merged['algorithm.surrogate_model.meta.name'] != baseline_algo]

    # Compute improvement (positive means better than ifbox)
    merged['improvement_over_ifbox'] = (
            merged['ifbox_anytime_performance'] - merged['anytime_performance']
    )
    return merged


def aggregate_improvement(df):
    # Aggregate median and quantiles of improvement per algorithm and step
    agg = (
        df.groupby(['benchmark.meta.name', 'algorithm.surrogate_model.meta.name', 'step'])
        .agg(
            q05_improvement=('improvement_over_ifbox', lambda x: x.quantile(0.05)),
            q95_improvement=('improvement_over_ifbox', lambda x: x.quantile(0.95)),
            median_improvement=('improvement_over_ifbox', 'median')
        )
        .reset_index()
    )
    return agg


def plot_per_task_improvement(agg_df):
    # Facet by benchmark, hue by algorithm
    g = sns.FacetGrid(
        agg_df,
        col='benchmark.meta.name',
        hue='algorithm.surrogate_model.meta.name',
        sharey=True,
        height=4,
        aspect=1.5
    )

    def plot_with_error(data, color, label, **kwargs):
        plt.plot(data['step'], data['median_improvement'], label=label, color=color)
        plt.fill_between(
            data['step'],
            data['q05_improvement'],
            data['q95_improvement'],
            color=color,
            alpha=0.2
        )

    g.map_dataframe(plot_with_error)
    g.add_legend()
    g.set_axis_labels("Fidelity (step)", "Improvement over ifbox (lower loss is better)")
    g.set_titles(col_template="{col_name}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line for no improvement
    plt.show()


if __name__ == '__main__':
    file = Path(
        '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/kisski_results/joint_results_bdd5b01.csv')
    df = pd.read_csv(file)
    import os
    import re

    # Fill missing algorithm names with 'ifbox'
    df['algorithm.surrogate_model.meta.name'] = df['algorithm.surrogate_model.meta.name'].fillna(
        value='ifbox')

    def extract_target_train_allocseed(s):
        # Regex: target_idx = digits, train_ids = anything inside brackets, allocation_seed = digits at end
        m = re.match(r'neps_root_directory_(\d+)_((?:\[.*?\]))_(\d+)_(\d+)$', s) # fixme: this will
        # have a split_seed before the allocation_seed in the current version!
        if m:
            return pd.Series({
                'target_task': m.group(1),
                'train_ids': m.group(2),
                'allocation_seed': m.group(3)
            })
        return pd.Series({'target_task': None, 'train_ids': None, 'allocation_seed': None})


    # Assuming df['filepath'] exists:
    df['parent_dir'] = df['filepath'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df[['target_task', 'train_ids', 'allocation_seed']] = df['parent_dir'].apply(
        extract_target_train_allocseed)

    # filter where target_idx is not None
    df = df[df['target_task'].notna()]

    # Compute anytime performance
    df_anytime = compute_anytime_performance(df)

    # Compute per-task improvement over ifbox
    df_improvement = compute_per_task_improvement(df_anytime, baseline_algo='ifbox')

    # Aggregate improvement statistics
    agg_improvement = aggregate_improvement(df_improvement)

    # Plot per-task improvement
    plot_per_task_improvement(agg_improvement)
