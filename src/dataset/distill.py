import warnings
from copy import deepcopy
from functools import partial
from typing import Union

import torch
from ifbo import BarDistribution, FTPFN
from ifbo.transformer import TransformerModel
from torch import nn
from tqdm import tqdm

from src.filelogger import BufferedFileLogger

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def constrain(x: torch.Tensor):
    """
    Constrain the input tensor to ensure that the first dimension is a differentiable integer
    (via straight-through estimator) and the remaining dimensions are constrained to the interval [0, 1]

    """
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

    assert constrained_tensor[:, :, 1:].min() >= 0, "Constrained tensor has negative values"
    assert constrained_tensor[:, :, 1:].max() <= 1, "Constrained tensor has values greater than 1"

    return constrained_tensor


class DistillContextTrainer:
    def __init__(self,
                 model: FTPFN,
                 optimizer: partial,
                 criterion: BarDistribution,
                 logger: BufferedFileLogger,
                 device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                 ):
        """

         Args:
        :param dataloader: DataLoader containing the training data.
        :param model: The model to be distilled.
        :param criterion:
        :param logger:
        """
        self.pfn = model
        self.model: TransformerModel = self.pfn.model
        self.device = device
        self.freeze_model(self.model, device)

        self.optimizer_partial = optimizer
        self.optimizer = None
        self.criterion = criterion

        self.logger = logger
        self.pbar = None

    @staticmethod
    def freeze_model(model, device):
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

    def test_on_new_task(self, context_x, context_y, task_x, task_y, context_sizes,
                         task_name, step=None, ):
        """
        Evaluate the distilled context on a new task.
        I.e. on the new task predict prepending the distilled context and then consecutively
        adding task context, evaluating at every step from 0 to len(task_y) - 1 by "by" steps.
        :param context_x: distilled context points (T x B x dim), T = number of tokens, B = batch size ==1, dim
         being (hp_index, fidelity, hyperparameter vector)
        :param context_y:
        :param task_x:
        :param task_y:
        :param task_name:
        :param by:
        :return:
        """
        context_x, context_y = context_x.to(self.device), context_y.to(self.device)
        task_x, task_y = task_x.to(self.device), task_y.to(self.device)

        losses = []

        for context_size in context_sizes:  # note how the final y is omitted here
            logits = self.model(
                (  # distilled context + observed x part of task, query for that task
                    torch.cat([context_x, task_x[:context_size], task_x[context_size:]],
                              dim=0),
                    # distilled labels + observed y part of task,
                    torch.cat([context_y, task_y[:context_size], ], dim=0)
                ),
                single_eval_pos=context_x.shape[0] + context_size
            )
            # y's associated with query for that task
            target = task_y[context_size:]

            loss = self.criterion(logits, target)
            loss = loss.view(-1, logits.shape[1])  # sometimes the seq length can be one off
            loss = loss.mean()

            self.logger.add_scalar(
                task_name,
                loss.item(),
                context_size, step
            )

        return losses

    def train(
            self,
            dataloader,
            x_init: torch.Tensor,
            y_init: torch.Tensor,

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

            n_steps: Number of distillation steps.
        """
        # map the fidelity and hp [0,1] to the logit space (which we optimize)

        x = deepcopy(x_init)
        y = deepcopy(y_init)

        p = x_init[:, :, 1:]
        # to enforce the [0,1] constraint, we first map the values that actually
        # live in the [0,1] space to the logit space (because during the loop we will apply sigmoid)
        x_init[:, :, 1:] = torch.log(p / (1 - p + 1e-6))

        distilled_points = nn.Parameter(x_init)
        distilled_labels = nn.Parameter(y_init)

        distilled_points = distilled_points.to(self.device)
        distilled_labels = distilled_labels.to(self.device)

        self.optimizer = self.optimizer_partial([distilled_points, distilled_labels])

        self.pbar = tqdm(range(int(n_steps)), desc="Distillation Progress", unit="step")
        for step in self.pbar:
            losses = []
            for x_query, y_query in dataloader:
                loss = self.train_step(x_query, y_query, distilled_points, distilled_labels)
                losses.append(loss)

                log.debug(
                    f"Step {step}: Loss: {loss.item()}, Distilled Points: {distilled_points.detach().sigmoid_()}")

            self.logger.add_scalar("train_loss", torch.mean(torch.stack(losses)).item(),
                                   distilled_points.shape[0], step)

            if step % val_log_frequency == 0:
                # testing on the same task, how well we predict the GT data given the distillation
                x_query, y_query = dataloader.dataset.get_shuffled_data()
                context_size = [x_query.shape[0]]
                reconstrucion_loss = self.test_on_new_task(
                    context_x=constrain(distilled_points).squeeze(1),
                    context_y=distilled_labels.view(-1),
                    task_x=x_query.squeeze(1),
                    task_y=y_query.view(-1),
                    context_sizes=context_size,
                    step=step,
                    task_name="nll_distilled_reconstruction"
                )

                self.pbar.set_postfix(
                    val_loss=reconstrucion_loss[0],
                    train_loss=torch.mean(torch.stack(losses)).item()
                )


        distilled_points = distilled_points.detach().cpu()
        distilled_points[:, :, 1:] = torch.sigmoid(distilled_points[:, :, 1:]).detach().cpu()

        if torch.allclose(x, distilled_points) or torch.allclose(y, distilled_labels):
            warnings.warn(
                "Distillation did not change the points. Check your optimizer and learning rate."
            )
        return distilled_points, distilled_labels

    def train_step(self, x_query, y_query, distilled_points, distilled_labels):
        # collate is messed up, batch dim is 1
        x_query = x_query.permute(1, 0, 2).to(self.device)  # T x B x dim
        y_query = y_query.permute(1, 0).to(self.device)  # T x B

        T, B, dim = x_query.shape

        # ENFORCE PARAMAMETER CONSTRAINTS ----------------------------------
        # The transformermodel places certain constraints on the fwd.
        constrained_tensor = constrain(distilled_points)

        # to ensure numerical stability:
        with torch.no_grad():
            constrained_tensor = torch.sigmoid(constrained_tensor).clamp(1e-3, 1 - 1e-3)

        # Get model predictions using distilled context
        logits = self.model(
            # ([x_train, x_query], ytrain)
            (  # notice, that x_query comes from the dataset and thus is constrained
                torch.cat([constrained_tensor.expand(-1, B, -1), x_query], dim=0),
                distilled_labels.expand(-1, B)
            ),
            single_eval_pos=constrained_tensor.shape[0]
        )

        # Compute the BarDistribution NLL loss
        loss = self.criterion(logits, y_query)
        # Found in PFNs4HPO.pfns4hpo.train.py: sometimes the seq length can be one off
        # that is because bar dist appends the mean
        loss = loss.view(-1, logits.shape[1])
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


if __name__ == '__main__':
    import ifbo
    from src.dataset.dataloading import DTrain
    from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch
    from src.dataset.dataloading import collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=2, single_eval_pos=500)

    sep = batch.single_eval_pos[0]
    x, y = batch.x[:sep], batch.y[:sep]

    # 0 task is the one we want to predict on
    target_task_x = x[:, 0:1]
    target_task_y = y[:, 0:1]

    x = x[:, 1:]
    y = y[:, 1:]
    print(x.shape, y.shape)

    # get random subsets of the dataset to be distilled for grad descent
    BATCH_SIZE = 100

    # communicate it to the dataset.
    dtrain = DTrain(x, y, length=BATCH_SIZE, sequence_length_max=500)
    x_query, y_query = dtrain[0]

    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 100  # FIXME: important hyperparameter
    x_init, y_init = dtrain.get_rnd_subset_initialization(DISTILL_SIZE)

    optimizer = partial(torch.optim.AdamW, lr=1e-3, weight_decay=0)

    collate_fn = partial(collate, min_length=10, max_length=500)
    dataloader = torch.utils.data.DataLoader(
        dtrain,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    logger = BufferedFileLogger(
        file_name="distill.csv",
        file_path=".",
        buffer_size=1000,
        header=("metric", "value", "context_size", "global_step"))

    trainer = DistillContextTrainer(
        model=ftpfn,
        optimizer=optimizer,
        criterion=pfn_backend.criterion,
        logger=logger,
        device=device
    )

    N_STEPS = 12
    distilled_x, distilled_y = trainer.train(
        dataloader=dataloader,
        x_init=x_init,
        y_init=y_init,
        n_steps=N_STEPS
    )

    # Plot distilled reconstruction loss. --------------------------------
    import matplotlib.pyplot as plt
    ax = logger.plot_scalar_curve(
        metric="nll_distilled_reconstruction",
        title="Distilled Reconstruction Loss",
        plot=False
    )
    ax = logger.plot_scalar_curve(
        metric="train_loss",
        title="Distilled Training Loss",
        plot=False,
        ax=ax
    )
    ax.set_xlabel("Context size")
    ax.ylabel("NLL Loss")
    ax.set_title("Distilled Reconstruction Loss")
    plt.show()

    # Plot downstream task performance with distilled context. ---------------
    CONTEXT_SIZES = range(10, target_task_x.shape[0], 20)
    losses = trainer.test_on_new_task(
        distilled_x, distilled_y,
        target_task_x, target_task_y,
        context_sizes=CONTEXT_SIZES
    )

    x_shape = distilled_x.shape
    baseline_x = torch.tensor([], device=device).view(0, x_shape[1], x_shape[2])
    baseline_y = torch.tensor([], device=device).view(0, x_shape[1])
    trainer.test_on_new_task(
        baseline_x, baseline_y,
        target_task_x, target_task_y,
        task_name='baseline (no distillation)',
        context_sizes=CONTEXT_SIZES
    )

    trainer.test_on_new_task(
        x, y,
        target_task_x, target_task_y,
        task_name='baseline (complete context task 0)',
        context_sizes=CONTEXT_SIZES
    )

    half_x = x.shape[0] // 2
    trainer.test_on_new_task(
        x[:half_x], y[:half_x],
        target_task_x, target_task_y,
        task_name='baseline (half context task 0)',
        context_sizes=CONTEXT_SIZES
    )

    ax = None
    for metric in ['incl. distilled context from task 0',
                   'baseline (no distillation)',
                   'baseline (complete context task 0)',
                   'baseline (half context task 0)']:
        ax = logger.plot_scalar_curve(
            metric=metric,
            plot=False,
            ax=ax,
        )

    ax.set_xlabel("Task's Context size")
    ax.set_ylabel("NLL Loss")
    ax.set_title("Distilled Context + increments of the new Task (nll)", )
    ax.legend()
    plt.show()
