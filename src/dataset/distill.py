import warnings
from copy import deepcopy
from functools import partial

import torch
from ifbo import BarDistribution, FTPFN
from ifbo.transformer import TransformerModel
from torch import nn
from tqdm import tqdm

from src.filelogger import BufferedFileLogger


class DTrain(torch.utils.data.Dataset):

    def __init__(self, x, y, batch_size=100, sequence_length_max=500):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.sequence_length_max = sequence_length_max

    def __len__(self):
        return self.batch_size

    def get_shuffled_data(self):
        shuffled_indices = torch.randperm(self.x.size(0))
        return self.x[shuffled_indices], self.y[shuffled_indices]

    def get_rnd_subset_initialization(self, size):
        """
        Get a random subset of the data for initialization.

        Args:
            size: The size of the subset to be returned.

        Returns:
            A random subset of the data.
        """
        x, y = self.get_shuffled_data()
        return x[:size], y[:size]

    def __getitem__(self, idx):
        # shuffle x and y in sequence dimension
        x_shuffled, y_shuffled = self.get_shuffled_data()

        # FIXME: is stochastisity maybe an important part here?
        # this does not work with the batching which requires equal size:
        # so maybe instead use shuffling with rnd src_masks?
        # size = torch.randint(10, self.sequence_length_max, (1,)).item()


        x = x_shuffled[:SIZE]
        y = y_shuffled[:SIZE]

        return x.squeeze(1), y.squeeze(1)


def constrain(x:torch.Tensor):
    # (0) straight-through estimator for integer constraint on hp index dimension
    quantized_dim0 = x[:, :, 0].round()
    ste_dim0 = quantized_dim0 - x[:, :, 0].detach() + x[:, :, 0]

    # (1) Apply sigmoid for interval constraint on fidelity & hp dimension
    constrained_dim1 = torch.sigmoid(x[:, :, 1:])

    # Combine dimensions while preserving gradients
    constrained_tensor = torch.cat([
        ste_dim0.unsqueeze(2),
        constrained_dim1,
    ], dim=-1)

    return constrained_tensor

def distill(
        dataloader,
        model: FTPFN,
        optimizer: partial,
        criterion,
        x_init: torch.Tensor,
        y_init: torch.Tensor,
        logger: BufferedFileLogger,
        n_steps=1e4,
        val_log_frequency=10
):
    """
    Ma et al 2024 In Context Data Distillation with TabPFN:

    Given a new dataset D_train and a query point x to classify, its class probability is
    computed as p θ(y|x, D_train), where D_train can be understood as the “prompt”. In-Context
    Distillation (ICD) – similarly to prompt-tuning (Lester et al., 2021)
    – optimizes the likelihood of the real data given the distilled one:
    L(D_train --> D_dist) = - E_{(x,y)~D_train)} [log p(y|x, D_dist)]
    Then, given a new point x to classify,  we use D_dist as the context and predict the label
    arg max y p θ(y|x, D_dist)

    Args:
        dataloader: DataLoader containing the training data.
        model: The model to be distilled.
        n_steps: Number of distillation steps.
    """
    # map the fidelity and hp [0,1] to the logit space (which we optimize)
    x = deepcopy(x_init)
    y = deepcopy(y_init)

    p = x_init[:, :, 1:]
    x_init[:, :, 1:] = torch.log(p / (1 - p + 1e-6))



    distilled_points = nn.Parameter(x_init)
    distilled_labels = nn.Parameter(y_init)

    distilled_points = distilled_points.to(device)
    distilled_labels = distilled_labels.to(device)

    optimizer = optimizer([distilled_points, distilled_labels])

    pfn = model
    model: TransformerModel = model.model
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    trajectory = []
    trajectory.append(distilled_points.detach().cpu().numpy())
    pbar = tqdm(range(int(n_steps)), desc="Distillation Progress", unit="step")
    for step in pbar:
        losses = []
        for x_query, y_query in dataloader:
            # collate is messed up, batch dim is 1
            x_query = x_query.permute(1, 0, 2).to(device)  # T x B x dim
            y_query = y_query.permute(1, 0).to(device)  # T x B

            T, B, dim = x_query.shape

            # ENFORCE PARAMAMETER CONSTRAINTS ----------------------------------
            # The transformermodel places certain constraints on the fwd.
            constrained_tensor = constrain(distilled_points)

            # Get model predictions using distilled context
            logits = model(
                # ([x_train, x_query], ytrain)
                ( # notice, that x_query comes from the dataset and thus is constrained
                    torch.cat([constrained_tensor.expand(-1, B, -1), x_query], dim=0),
                    distilled_labels.expand(-1, B)
                ),
                single_eval_pos=constrained_tensor.shape[0]
            )

            # Compute the BarDistribution NLL loss
            loss = criterion(logits, y_query)
            # Found in PFNs4HPO.pfns4hpo.train.py: sometimes the seq length can be one off
            # that is because bar dist appends the mean
            loss = loss.view(-1, logits.shape[1])
            loss = loss.mean()

            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # print(distilled_points[0, 0].detach().sigmoid_())

            if True:
                trajectory.append(deepcopy(distilled_points.detach().cpu().numpy()))

        logger.add_scalar("train_loss", torch.mean(torch.stack(losses)).item(), step)

        if step % val_log_frequency == 0:
            # Log the loss every 100 steps
            x_query, y_query = dataloader.dataset.get_shuffled_data()
            logits = pfn(x_train=constrain(distilled_points).squeeze(1),
                           y_train=distilled_labels.view(-1), x_test=x_query.squeeze(1))
            loss = criterion(logits, y_query.view(-1))
            loss = loss.view(-1, logits.shape[1])  # sometimes the seq length can be one off
            # that is because bar dist appends the mean
            loss = loss.mean()
            logger.add_scalar("val_loss", loss.item(), step)
            pbar.set_postfix(val_loss=loss.item())


    trajectory = torch.tensor(trajectory)
    trajectory[:, :, :, 1:] = trajectory[:, :, :, 1:].sigmoid_()
    # since the original points are not enforced to live on the scale

    distilled_points = distilled_points.detach().cpu()
    distilled_points[:, :, 1:] = torch.sigmoid(distilled_points[:, :, 1:]).detach().cpu()

    logger.trajectory = trajectory

    if torch.allclose(x, distilled_points):
        warnings.warn("Distillation did not change the points. Check your optimizer and learning rate.")
    return distilled_points, distilled_labels


def plot_optimization_trajectory(trajectory):
    """
    Plots the optimization trajectory of the distilled points in 2D.

    Args:
        trajectory: A list or numpy array containing the positions of distilled points at each step.
                    Shape: (n_steps, n_points, dimensions)
                    Only first two dimensions are plotted.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))

    trajectory = trajectory.squeeze(2)

    n_steps = len(trajectory)

    # Plot all trajectories
    for point_idx in range(trajectory.shape[1]):
        plt.plot(
            trajectory[:, point_idx, 1],
            trajectory[:, point_idx, 2],
            label=f"Point {point_idx}"
        )

    plt.scatter(
        trajectory[0, :, 1],
        trajectory[0, :, 2],
        c='red',
        label='Start',
        marker='o'
    )

    plt.scatter(
        trajectory[-1, :, 1],
        trajectory[-1, :, 2],
        c='green',
        label='End',
        marker='x'
    )

    plt.title("Optimization Trajectory of Distilled Points")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)

    plt.show()




if __name__ == '__main__':
    import ifbo

    from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=1, single_eval_pos=500)

    sep = batch.single_eval_pos[0]
    x, y = batch.x[:sep], batch.y[:sep]
    print(x.shape, y.shape)

    # get random subsets of the dataset to be distilled for grad descent
    BATCH_SIZE = 100
    SIZE = 50 # FIXME: we should randomly sample this in the collate of the dataloader and
    # communicate it to the dataset.
    dtrain = DTrain(x, y, batch_size=BATCH_SIZE, sequence_length_max=500)
    x_query, y_query = dtrain[0]

    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 10
    x_init, y_init = dtrain.get_rnd_subset_initialization(DISTILL_SIZE)

    optimizer = partial(torch.optim.AdamW, lr=1e-3, weight_decay=0)
    dataloader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True)
    logger = BufferedFileLogger(file_name="distill.csv", file_path=".", buffer_size=1000,
                                header=("metric", "value", "global_step"))

    distilled_x, distilled_y = distill(
        dataloader,
        ftpfn,
        optimizer=optimizer,
        criterion=pfn_backend.criterion,
        logger=logger,
        x_init=x_init,
        y_init=y_init,
        n_steps=100
    )
    logger._flush()
    logger.plot_scalar_curve(metric="val_loss", title="val_loss", )
    plot_optimization_trajectory(logger.trajectory)

