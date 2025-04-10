from functools import partial

import torch
from ifbo import BarDistribution, FTPFN
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
        size = 101
        x = x_shuffled[:size]
        y = y_shuffled[:size]

        return x, y


def distill(
        dataloader,
        model: FTPFN,
        optimizer: partial,
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
    distilled_points = nn.Parameter(x_init)
    distilled_labels = nn.Parameter(y_init)

    distilled_points = distilled_points.to(device)
    distilled_labels = distilled_labels.to(device)

    optimizer = optimizer([distilled_points, distilled_labels])

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    criterion: [BarDistribution] = model.model.criterion

    for step in tqdm(range(int(n_steps))):
        for x_query, y_query in dataloader:
            # collate is messed up, batch dim is 1
            x_query = x_query.squeeze(-2).permute(1, 0, 2) # T x B x dim
            y_query = y_query.squeeze(-1).permute(1, 0)  # T x B

            x_query = x_query.to(device)
            y_query = y_query.to(device)

            T, B, dim = x_query.shape

            # Get model predictions using distilled context
            # fixme: while the shapes are ok, the values of x_train, x_test
            #  seem to wrongly ordered (in terms of columns), because FTPFN._check_input fails
            # x_train[:, 1].max() > 1 --> True seems to be the idx tensor.
            # x_test[:, 1].max() > 1
            #FIXME: we may also need to save guard here, i.e. clip the updates or do a sigmoid on
            # them
            logits = model(
                # expand the distilled points as shared object for the batch
                x_train=distilled_points.expand(-1, B, -1),
                y_train=distilled_labels.expand(-1, B),
                x_test=x_query)

            # Compute the BarDistribution NLL loss
            loss = criterion(logits, y_query)
            # Found in PFNs4HPO.pfns4hpo.train.py: sometimes the seq length can be one off
            # that is because bar dist appends the mean
            loss = loss.view(-1, logits.shape[1])
            logger.add_scalar("train_loss", loss.item(), step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % val_log_frequency == 0:
            # Log the loss every 100 steps
            x_query, y_query = dataloader.dataset.get_shuffled_data()
            logits = model(x_train=distilled_points, y_train=distilled_labels, x_test=x_query)
            loss = criterion(logits, y_query)
            loss = loss.view(-1, logits.shape[1])  # sometimes the seq length can be one off
            # that is because bar dist appends the mean
            logger.add_scalar("val_loss", loss.item(), step)

    return distilled_points, distilled_labels


if __name__ == '__main__':
    import ifbo

    from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ifbo.surrogate.FTPFN(version="0.0.1", device=device)

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=1, single_eval_pos=500)

    sep = batch.single_eval_pos[0]
    x, y = batch.x[:sep], batch.y[:sep]
    print(x.shape, y.shape)

    # get random subsets of the dataset to be distilled for grad descent
    BATCH_SIZE = 100
    dtrain = DTrain(x, y, batch_size=BATCH_SIZE, sequence_length_max=500)
    x_query, y_query = dtrain[0]

    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 10
    x_init, y_init = dtrain.get_rnd_subset_initialization(DISTILL_SIZE)

    optimizer = partial(torch.optim.AdamW, lr=1e-4, weight_decay=0)
    dataloader = torch.utils.data.DataLoader(dtrain, batch_size=BATCH_SIZE, shuffle=True)
    logger = BufferedFileLogger(file_name="distill.csv", file_path=".", buffer_size=1000,
                                header=("metric", "value", "global_step"))

    distilled_x, distilled_y = distill(dataloader, model, optimizer=optimizer, logger=logger,
                                       x_init=x_init, y_init=y_init, n_steps=1e4)
