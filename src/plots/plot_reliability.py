if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns
    from pathlib import Path

    FILE = Path(__file__).resolve().parents[
               2] / 'luis_results/pfnimpute-1sttrial/joint_results_reliability_d51cfa3.csv'

    df = pd.read_csv(FILE, index_col=0)

    # Filter for a single benchmark and algorithm combination
    single_combination_df = df[
        (df['benchmark.meta.name'] == 'taskset') &
        (df['algorithm.surrogate_model.meta.name'] == 'pfn_impu') &
        (df['train_ids'] == '[16, 8, 6, 17, 4]') &
        (df['target_task'] == 11)
        ]

    # Plot all tasks in a single plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=single_combination_df,
        x='context_size',
        y='value',
        hue='task',
        marker='o'
    )
    plt.xlabel('Context Size')
    plt.ylabel('Reliability Score')
    plt.title('Reliability Score by Context Size for All Tasks (Single Combination)')
    plt.legend(title='Task')
    plt.show()

    # --------------------------------------------------------------------------

    # Filter and prepare data
    df = df[df['metric'] == 'reliability_score'].copy()
    df['context_size'] = pd.to_numeric(df['context_size'], errors='coerce')
    df = df.dropna(subset=['context_size', 'value'])
    df = df.sort_values('context_size')

    # Facet by benchmark (columns) and algorithm (rows), hue by task
    g = sns.FacetGrid(
        df,
        col='benchmark.meta.name',
        row='algorithm.surrogate_model.meta.name',
        hue='task',
        margin_titles=True,
        height=4,
        aspect=1.5
    )
    g.map(sns.lineplot, 'context_size', 'value', marker='o')
    g.set_axis_labels('Context Size', 'Reliability Score')
    g.add_legend(title='Task')
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    plt.show()

    #
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # # Assume df is your DataFrame
    #
    # # Use relplot for convenience, which wraps FacetGrid and lineplot
    # g = sns.relplot(
    #     data=df,
    #     x="context_size",
    #     y="value",
    #     kind="line",
    #     row="target_task",
    #     col="train_ids",
    #     hue="task",  # or swap hue and style as needed
    #
    #     style="benchmark.meta.name",  # if you want different line styles for each task
    #     facet_kws={'margin_titles': True},
    #     height=3,
    #     aspect=1.5
    # )
    #
    # g.set_titles(row_template='{row_name}', col_template='{col_name}')
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle('Value vs. Context Size Faceted by Target Task, Train IDs, and Benchmark')
    # plt.show()
