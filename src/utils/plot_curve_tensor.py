import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_curve_tensor(X, Y, single_eval_pos, idx=4, ax=None, show=True):
    """
    Plots fidelity vs y values colored by unique configuration curves

    Parameters:
    X (array): Input array of shape (n_samples, n_features, n_parameters)
    Y (array): Target array of shape (n_samples, n_features)
    single_eval_pos (array): Positions array for slicing
    idx (int): Index for specific feature analysis (default=4)

    Returns:
    pd.DataFrame: Processed dataframe for inspection
    """
    # Create DataFrame with fidelity and config parameters
    df = pd.DataFrame(
        X[:single_eval_pos[idx], idx, 1:],
        columns=['fidelity'] + [f'cfg{i}' for i in range(1, X[:, idx, 1:].shape[1])]
    )

    # Transform y-values
    df['y'] = Y[:single_eval_pos[idx], idx]

    # Create unique curve identifiers
    config_cols = [col for col in df.columns if col.startswith('cfg')]
    df['curve_id'] = df[config_cols].apply(lambda row: hash(tuple(row)), axis=1)
    df['curve_id'] = pd.factorize(df['curve_id'])[0]  # Get factorized codes

    # Generate plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        ax=ax,
        data=df,
        x='fidelity',
        y='y',
        hue='curve_id',
        palette='tab10',
        legend='full'
    )
    ax.set_title(f'Fidelity vs y (Index {idx}), Colored by Configuration')

    if show:
      plt.show()


