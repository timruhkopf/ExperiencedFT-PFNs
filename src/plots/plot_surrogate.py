"""
df = pd.read_csv('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/surrogate_joint_results.csv')
df.columns
Out[5]:
Index(['Unnamed: 0', 'metric', 'value', 'global_step', 'context_size', 'task',
'target_task', 'train_ids', 'seed', 'experiment_name',
'model.meta.name', 'benchmark.meta.name', 'file_path'],
dtype='object')


df['metric'].unique()
Out[6]:
array(['train_loss', 'nll_distilled_reconstruction',
'nll_distilled_predictive', 'distill',
'Baseline: approx. conditioning on complete related',
'Baseline: naked pfn', 'x_init context (no-distillation)',
'reliability_score', 'pfn_softmax', 'pfn_argmin'], dtype=object)

where i only want these metrics:


'Baseline: approx. conditioning on complete related',
'Baseline: naked pfn', 'x_init context (no-distillation)',
'pfn_softmax', 'pfn_argmin'

for every context_size, task, target_task, train_ids, seeds, benchmark,
i would like to have a ranking for each metric on the value (lower is better)

after that, i would want to plot the mean rank against context_size stratified by benchmark and for all methods
"""
import pandas as pd
df = pd.read_csv('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results'
                 '/surrogate_joint_results.csv', engine='pyarrow')
df1 = pd.read_csv('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/joint_results'
                  '.csv',)

df = pd.concat([df, df1], axis=0)

df = df.drop(columns=['Unnamed: 0'])

metrics_of_interest = [
    'Baseline: naked pfn',
    'pfn_softmax',
    'pfn_argmin',
    'Baseline: approx. conditioning on complete related',
    'x_init context (no-distillation)',
]
# fixme:  the baselines are recorded per meta-task, so we should actually search for the best one.
#     'Baseline: approx. conditioning on complete related',
#     'x_init context (no-distillation)',

filtered_df = df[df['metric'].isin(metrics_of_interest)]

group_cols = [
    'context_size', 'task', 'target_task', 'train_ids', 'seed', 'benchmark.meta.name'
]

dedup_metrics = [
    'Baseline: naked pfn',
    'Baseline: approx. conditioning on complete related',
    'x_init context (no-distillation)'
]


dedup_df = (
    filtered_df[filtered_df['metric'].isin(dedup_metrics)]
    .sort_values('value')
    .groupby(group_cols + ['metric'], as_index=False)
    .first()
)

non_dedup_df = filtered_df[~filtered_df['metric'].isin(dedup_metrics)]
filtered_df_dedup = pd.concat([dedup_df, non_dedup_df], ignore_index=True)


#
# filtered_df_dedup.to_csv('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/surrogate_joint_results_dedup.csv')
# filtered_df_dedup = pd.read_csv('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/surrogate_joint_results_dedup.csv')
filtered_df_dedup['rank'] = (
    filtered_df_dedup.groupby(group_cols)['value']
    .rank(method='min', ascending=True)
)



# # lower values
# filtered_df['rank'] = filtered_df.groupby(group_cols)['value'].rank(method='min', ascending=True)


agg_df = (
    filtered_df_dedup
    .groupby(['context_size', 'benchmark.meta.name', 'metric'])['rank']
    .mean()
    .reset_index()
)

import seaborn as sns
import matplotlib.pyplot as plt

# Use relplot for faceting by benchmark
g = sns.relplot(
    data=agg_df,
    x='context_size',
    y='rank',
    hue='metric',
    kind='line',
    col='benchmark.meta.name',   # Facet by benchmark
    col_wrap=3,                  # Adjust depending on number of benchmarks
    marker='o',
    height=4,
    aspect=1.2
)

g.set_axis_labels('Context Size', 'Mean Rank (lower is better)')
g.set_titles('{col_name}')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Mean Rank vs. Context Size by Method')
plt.show()
