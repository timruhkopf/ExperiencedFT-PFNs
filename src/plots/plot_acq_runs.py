import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


def compute_anytime_performance(df):
    # Sort by filepath and epoch to ensure correct order
    df = df.sort_values(['filepath', 'epoch'])
    # Compute anytime performance (cumulative min loss) for each run
    df['anytime_performance'] = df.groupby('filepath')['loss'].cummin()

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


def plot_anytime_performance(agg_df):
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
    plt.show()


if __name__ == '__main__':
    file = Path(
        '/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/kisski_results/2025-05-14_results.csv')
    df = pd.read_csv(file)

    df['benchmark.meta.name'].unique()

    df['algorithm.surrogate_model.meta.name'].unique()

    df.columns

    df['algorithm.surrogate_model.meta.name'] = df[
        'algorithm.surrogate_model.meta.name'].fillna(value='ifbox')

    # df is your original DataFrame
    df_anytime = compute_anytime_performance(df)
    agg_df = aggregate_anytime(df_anytime)
    plot_anytime_performance(agg_df)
