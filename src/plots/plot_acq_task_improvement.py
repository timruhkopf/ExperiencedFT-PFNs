import os
import re
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


def compute_per_task_improvement(df_anytime, baseline_algo='ifbo'):
    # Identify the baseline (ifbo) performance per task and step
    baseline_df = df_anytime[df_anytime['algorithm.surrogate_model.meta.name'] == baseline_algo]
    baseline_df = baseline_df[
        ['train_ids', 'target_task',
         # 'split_seed',
         'step', 'anytime_performance']
    ].rename(columns={'anytime_performance': 'ifbo_anytime_performance'})

    # repetitions over the baseline; so we want the median over ifbo
    baseline_df = (
        baseline_df.groupby(['train_ids', 'target_task', 'step'])       # 'split_seed',
        .agg(ifbo_anytime_performance=('ifbo_anytime_performance', 'median'))
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

    # Compute improvement (positive means better than ifbo)
    merged['improvement_over_ifbo'] = (
            merged['ifbo_anytime_performance'] - merged['anytime_performance']
    )
    return merged


def aggregate_improvement(df):
    # Aggregate median and quantiles of improvement per algorithm and step
    agg = (
        df.groupby(['benchmark.meta.name', 'algorithm.surrogate_model.meta.name', 'step'])
        .agg(
            q05_improvement=('improvement_over_ifbo', lambda x: x.quantile(0.05)),
            q95_improvement=('improvement_over_ifbo', lambda x: x.quantile(0.95)),
            median_improvement=('improvement_over_ifbo', 'median')
        )
        .reset_index()
    )
    return agg


def plot_per_task_improvement(agg_df, save_path):
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
    g.set_axis_labels("Fidelity (step)", "Improvement over ifbo (lower loss is better)")
    g.set_titles(col_template="{col_name}")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line for no improvement
    if save_path:
        g.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_anytime_performance(df, save_path=None):
    """Plot and optionally save anytime performance visualization"""
    g = sns.relplot(
        data=df,
        x='step',
        y='anytime_performance',
        hue='algorithm.surrogate_model.meta.name',
        col='target_task',
        kind='line',
        markers=False,
        dashes=False
    )

    if save_path:
        g.savefig(save_path)
        plt.close()
    else:
        plt.show()

def extract_target_train_allocseed(s, pattern=r'neps_root_directory_(\d+)_(\d+)_(\d+)_(\d+)$'):
    # Regex: target_idx = digits, train_ids = anything inside brackets, allocation_seed = digits at end
    m = re.match(pattern, s)
    # have a split_seed before the allocation_seed in the current version!
    if m:
        return pd.Series({
            'target_task': m.group(1),
            'train_ids': m.group(2),
            'allocation_seed': m.group(3)
        })
    return pd.Series({'target_task': None, 'train_ids': None, 'allocation_seed': None})


def main(
        csv_file: str,
        plot_type: str = "improvement",
        baseline_algo: str = "ifbo",
        save_path: str = None,
        algorithms: str = "ifbo,pfn_impute"
):
    """
    Command-line interface for performance visualization

    Args:
        csv_file: Path to input CSV file
        plot_type: Type of plot to generate (improvement|anytime)
        baseline_algo: Baseline algorithm for comparison
        save_path: Optional path to save plot (default shows interactive)
        algorithms: Comma-separated list of algorithms to include
    """
    df = pd.read_csv(Path(csv_file))
    df['algorithm.surrogate_model.meta.name'] = df['algorithm.surrogate_model.meta.name'].fillna(
        'ifbo')

    # Filter algorithms
    valid_algorithms = [a.strip() for a in algorithms.split(",")]
    df = df[df['algorithm.surrogate_model.meta.name'].isin(valid_algorithms)]

    # Data processing pipeline
    df['parent_dir'] = df['filepath'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df.loc[:,['target_task', 'train_ids', 'allocation_seed']] = df['parent_dir'].apply(
        lambda x: extract_target_train_allocseed(x)
    )
    df = df[df['target_task'].notna()]

    df_anytime = compute_anytime_performance(df)

    # Plot selection
    if plot_type == "improvement":
        df_improvement = compute_per_task_improvement(df_anytime, baseline_algo)
        agg_improvement = aggregate_improvement(df_improvement)
        plot_per_task_improvement(agg_improvement, save_path)
    elif plot_type == "anytime":
        plot_anytime_performance(df_anytime, save_path)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}. Use 'improvement' or 'anytime'")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
