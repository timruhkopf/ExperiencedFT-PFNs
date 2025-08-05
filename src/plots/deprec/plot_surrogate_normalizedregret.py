import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_normalized_regret(df):
    """Calculate normalized regret [0,1] per group"""
    group_cols = ['context_size', 'task', 'target_task', 'train_ids', 'seed', 'benchmark.meta.name']

    # Get min/max per group
    min_max = df.groupby(group_cols)['value'].agg(['min', 'max']).reset_index()

    # Merge and calculate
    df = df.merge(min_max, on=group_cols, how='left')
    df['normalized_regret'] = (df['value'] - df['min']) / (df['max'] - df['min'])
    df['normalized_regret'] = df['normalized_regret'].fillna(0)  # Handle min=max cases

    return df.drop(columns=['min', 'max'])


# --- Main Pipeline ---
if __name__ == "__main__":
    # 1. Load data
    import pandas as pd

    files = [
        '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/surrogate_joint_results.csv',
        '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/joint_results.csv',
        '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/joint_results_hung.csv'
    ]


    df = pd.concat([pd.read_csv(f) for f in files], axis=0)

    df = df.drop(columns=['Unnamed: 0'])
    # 2. Filter metrics
    metrics_of_interest = [
        'Baseline: naked pfn',
        'pfn_softmax',
        'pfn_argmin',
        'Baseline: approx. conditioning on complete related',
        'x_init context (no-distillation)'
    ]
    filtered_df = df[df['metric'].isin(metrics_of_interest)].copy()

    # 3. Deduplicate baselines (keep best per group)
    dedup_metrics = [
        'Baseline: naked pfn',
        'Baseline: approx. conditioning on complete related',
        'x_init context (no-distillation)'
    ]
    groups = ['context_size', 'task', 'target_task', 'train_ids', 'seed', 'benchmark.meta.name']
    dedup_idx = filtered_df[filtered_df['metric'].isin(dedup_metrics)] \
        .groupby([*groups, 'metric'])['value'] \
        .idxmin()
    filtered_df = pd.concat([
        filtered_df.loc[dedup_idx],
        filtered_df[~filtered_df['metric'].isin(dedup_metrics)]
    ])

    # 4. Compute metrics
    df_with_regret = compute_normalized_regret(filtered_df)
    df_with_regret['rank'] = df_with_regret.groupby(
        groups)[
        'value'] \
        .rank(method='min', ascending=True)

    # 5. Aggregate results
    agg_df = df_with_regret.groupby(['context_size', 'benchmark.meta.name', 'metric']) \
        .agg(mean_rank=('rank', 'mean'),
             mean_regret=('normalized_regret', 'mean')) \
        .reset_index()

    # 6. Plot results
    g = sns.relplot(
        data=agg_df,
        x='context_size',
        y='mean_regret',
        hue='metric',
        kind='line',
        col='benchmark.meta.name',
        col_wrap=3,
        marker='o',
        height=4,
        aspect=1.2
    )
    g.set_axis_labels('Context Size', 'Normalized Regret')
    g.set_titles('{col_name}')
    g.fig.suptitle('Performance by Context Size and Benchmark', y=1.02)
    plt.show()
