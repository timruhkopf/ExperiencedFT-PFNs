import torch
import hashlib

from dataset.deprec.taskprior import MetaTaskPriorSameProblem


def incumbent_max(Y):
    # Create cumulative max tensor
    cum_max, _ = torch.cummax(Y, dim=1)

    # Create mask for positions where current value equals the running max
    mask = (Y == cum_max)

    # Zero out non-max positions and return
    return cum_max * mask


uniform_auc = lambda y: 0.5 * (y[:, 0] + y[:, -1]) + y[:, 1:-1].sum(dim=1)


def sample_wrapper(args):
    import torch
    import numpy as np
    alpha, context_size, prior_instance, repetitions = args
    # Ensure thread-safe RNG initialization
    np.random.seed()  # Different seed for each process
    Y = []
    for _ in range(repetitions):
        _, y = prior_instance.sample_from_task(alpha=alpha, context_size=context_size)
        Y.append(y)

    return torch.stack(Y, dim=0).cpu(), prior_instance.__hash__()
    # model_weights_hash(prior_instance.relation_prior.model)


def model_weights_hash(model: torch.nn.Module) -> str:
    """Generate SHA256 hash of model parameters"""
    hasher = hashlib.sha256()

    for param in sorted(model.parameters(), key=lambda p: p.data_ptr()):
        # Hash parameter data and requires_grad state
        hasher.update(param.data.cpu().numpy().tobytes())
        hasher.update(bytes([param.requires_grad]))

    return hasher.hexdigest()


if __name__ == '__main__':
    import torch

    meta_task_prior = MetaTaskPriorSameProblem(
        dim_hyperparameters=3,

        n_fidelities=100,
        seq_len=1000,
        device='cpu'
    )

    # Sample a batch of related tasks
    device = 'cpu'
    sample_size = 1000
    # prior: 10 ** np.random.uniform(-4, -1)
    # high alpha --> uniform; explorative
    # low alpha --> more concentrated, exploitative
    from multiprocessing import Pool, set_start_method
    import numpy as np

    # Required for safe multiprocessing on some systems
    set_start_method('spawn', force=True)

    # Parallel sampling for each alpha
    alphas = [-4, -3.5, -3, -2.5, -2, -1.5, -1]
    # for alpha in [10 ** e for e in alphas]:
    with Pool() as pool:
        results = pool.map(
            sample_wrapper,
            [(10 ** alpha, 500, meta_task_prior, sample_size) for alpha in alphas]
        )

    # FIXME: The hashes are not the same for different alpha values
    for res in results:
        print(res[1])

    # plot the kernel density estimation of the AUC
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    Ys = [res[0] for res in results]
    for alpha, y in zip(alphas, Ys ):
        incumbent = incumbent_max(y)
        auc = uniform_auc(incumbent)
        sns.kdeplot(auc.cpu().numpy(), label=f"alpha=10**{alpha}", fill=False)
    plt.xlabel("AUC")

    plt.ylabel("Density")
    plt.title("Kernel Density Estimation of AUC reward")

    plt.legend()
    plt.show()
