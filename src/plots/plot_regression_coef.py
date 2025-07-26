import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_reliability_regression_coef(file_path: str):


    # Load the data
    df = pd.read_csv(file_path)
    df = df[df['metric'] =='projection/beta']
    del df['metric']

    # Suppose df is your DataFrame
    # Select columns that start with 'beta_'
    beta_cols = [col for col in df.columns if col.startswith('beta_')]

    # # Plot each beta column vs 'step'
    for col in beta_cols:
        sns.lineplot(df, x='step', y=col, label=col)

    plt.xlabel('step')
    plt.ylabel('beta values')
    plt.legend()
    plt.show()

def plot_nll_scores(file_path: str):
    # Load the data
    df = pd.read_csv(file_path)
    df = df[df['metric'] =='nll']
    del df['metric']

    # Plot each nll column vs 'step'
    nll_cols = [col for col in df.columns if 'nll' in col ]
    for col in nll_cols:
        sns.lineplot(df, x='step', y=col, label=col)

    plt.xlabel('step')
    plt.ylabel('NLL scores')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    import fire
    fire.Fire(
        {
            'plot_reliability_regression_coef': plot_reliability_regression_coef,
            'plot_nll_scores': plot_nll_scores
        }
    )
