from src.dataset.taskprior import MetaTaskPriorSameProblem


def incumbent_max(Y):
    # Create cumulative max tensor
    cum_max, _ = torch.cummax(Y, dim=1)

    # Create mask for positions where current value equals the running max
    mask = (Y == cum_max)

    # Zero out non-max positions and return
    return cum_max * mask

uniform_auc = lambda y: 0.5 * (y[:, 0] + y[:, -1]) + y[:, 1:-1].sum(dim=1)

if __name__ == '__main__':
    import torch

    meta_task_prior = MetaTaskPriorSameProblem(
        dim_hyperparameters=3,

        n_fidelities=100,
        seq_len=1000,
        device='cpu'
    )



    # plot the kernel density estimation of the AUC
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # Sample a batch of related tasks
    device = 'cpu'
    sample_size = 500
    # prior: 10 ** np.random.uniform(-4, -1)
    # high alpha --> uniform; explorative
    # low alpha --> more concentrated, exploitative
    for alpha in [10 ** e for e in [-4, -3.5, -3, -2.5, -2, -1.5, -1]]:
        X, Y = [], []
        for _ in range(sample_size):
            x, y = meta_task_prior.sample_from_task(alpha=alpha, context_size=500)
            # X.append(x)
            Y.append(y)

        # X = torch.stack(X, dim=1).to(device).float()
        Y = torch.stack(Y, dim=1).to(device).float()

        incumbent = incumbent_max(Y.T)

        auc = uniform_auc(incumbent)

        sns.kdeplot(auc.cpu().numpy(), label=f"alpha={alpha}", fill=False)
    plt.xlabel("AUC")

    plt.ylabel("Density")
    plt.title("Kernel Density Estimation of AUC reward")

    plt.legend()
    plt.show()



