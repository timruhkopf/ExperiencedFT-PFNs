import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import fire


def compute_anytime_performance(df, minimize=True):
    # Sort by filepath and epoch to ensure correct order
    df = df.sort_values(['filepath', 'epoch'])
    # Compute anytime performance (cumulative min loss) for each run
    if minimize:
        df['anytime_performance'] = df.groupby('filepath')['loss'].cummin()
    else:
        df['anytime_performance'] = df.groupby('filepath')['loss'].cummax()

    # add arange ids to the dataframe
    df['step'] = df.groupby('filepath').cumcount()
    return df


def aggregate_anytime(df):
    agg = (
        df.groupby(['benchmark.meta.name', 'algorithm.surrogate_model.meta.name', 'step'])
        .agg(
            q05_anytime=('anytime_performance', lambda x: x.quantile(0.05)),
            q95_anytime=('anytime_performance', lambda x: x.quantile(0.95)),
            median_anytime=('anytime_performance', 'median')
        )
        .reset_index()
    )
    return agg


def plot_anytime_performance(agg_df, title=None):
    # Facet by benchmark, hue by algorithm
    g = sns.FacetGrid(
        agg_df,
        col='benchmark.meta.name',
        hue='algorithm.surrogate_model.meta.name',
        sharey=False,
        height=4,
        aspect=1.5
    )

    # Plot mean and std as line and fill
    def plot_with_error(data, color, label, **kwargs):
        plt.plot(data['step'], data['median_anytime'], label=label, color=color)
        plt.fill_between(
            data['step'],
            data['q05_anytime'],
            data['q95_anytime'],
            color=color,
            alpha=0.2
        )

    g.map_dataframe(plot_with_error)
    g.add_legend()
    g.set_axis_labels("Fidelity (step)", "Anytime Performance (Mean Loss)")
    g.set_titles(col_template="{col_name}")
    if title is not None:
        plt.title(title)

    plt.show()

    return g




def main(
    file='/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/fixed_reliability'
         '/fix_joint_results_94dc71b.csv',
    minimize=True,
    save=True,
        title=None
):
    file = Path(file)
    df = pd.read_csv(file)

    if 'epoch' not in df.columns:
        df['epoch'] = df.index

    print('Unique benchmarks:', df['benchmark.meta.name'].unique())
    print('Unique surrogate models:', df['algorithm.surrogate_model.meta.name'].unique())
    print('Columns:', df.columns)

    df['algorithm.surrogate_model.meta.name'] = df['algorithm.surrogate_model.meta.name'].fillna(value='ifbo')

    # df is your original DataFrame
    df_anytime = compute_anytime_performance(df, minimize=minimize)
    agg_df = aggregate_anytime(df_anytime)
    g = plot_anytime_performance(agg_df, title=title)

    if save:
        output_file = file.parent / f"anytime_performance_{file.stem}.png"
        g.savefig(output_file)
        print(f"Plot saved to {output_file}")

if __name__ == '__main__':
    fire.Fire(main)
#
#
# if __name__ == '__main__':
#     file = Path(
#         '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/luis_results/fixed_reliability/fix_joint_results_94dc71b.csv')
#     df = pd.read_csv(file)
#
#     df['benchmark.meta.name'].unique()
#
#     df['algorithm.surrogate_model.meta.name'].unique()
#
#     df.columns
#
#     df['algorithm.surrogate_model.meta.name'] = df[
#         'algorithm.surrogate_model.meta.name'].fillna(value='ifbo')
#
#     # df is your original DataFrame
#     df_anytime = compute_anytime_performance(df)
#     agg_df = aggregate_anytime(df_anytime)
#     plot_anytime_performance(agg_df)
