"""Quick and dirty code for interactive plotting in debug.
TODO make this a proper script."""
if __name__ == '__main__':

    # self is class

    df = self.mixture_logger.dataframe
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Filter rows where metric == 'reliability'
    df_rel = df[df['metric'] == 'reliability']
    # Select columns to plot
    cols_to_plot = ['target_reliability'] + [col for col in df.columns if
                                             col.startswith('related_reliability_')]
    # Melt dataframe to long format for seaborn
    df_long = df_rel.melt(id_vars='step', value_vars=cols_to_plot,
                          var_name='reliability_type', value_name='reliability_value')
    # Plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_long, x='step', y='reliability_value', hue='reliability_type')
    plt.title('Weight vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Softmax weight')
    plt.show()