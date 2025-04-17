import warnings
from copy import deepcopy
from functools import partial
from typing import Union, Tuple

import torch
from ifbo import BarDistribution, FTPFN
from ifbo.transformer import TransformerModel
from torch import nn
from tqdm import tqdm

from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.filelogger import BufferedFileLogger

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class AdaptiveTemperature(nn.Module):
    def __init__(self, init_temp=4.0):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(init_temp))

    def forward(self, x):
        return torch.sigmoid(x / self.temp.clamp(min=0.5, max=4.0))


def constrain(x: torch.Tensor, temp):
    """
    Constrain the input tensor to ensure that the first dimension is a differentiable integer
    (via straight-through estimator) and the remaining dimensions are constrained to the interval [0, 1]

    """
    # (0) straight-through estimator for integer constraint on hp index dimension
    # quantized_dim0 = x[:, :, 0].round()
    # ste_dim0 = quantized_dim0 - x[:, :, 0].detach() + x[:, :, 0]

    # temperature-annealed; straight-through
    soft_dim0 = torch.sigmoid(x[:, :, 0] / temp)
    quantized_dim0 = soft_dim0.round()
    ste_dim0 = quantized_dim0 - soft_dim0.detach() + soft_dim0

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
                 temperature_annealing: AdaptiveTemperature = None
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

        self.evaluator = TestOnNewTaskNLL(self.criterion, self.logger, self.device)
        self.test_on_new_task = self.evaluator.test_on_new_task

        if temperature_annealing is None:
            self.temperature_annealing = AdaptiveTemperature()
        else:
            self.temperature = 1

    @staticmethod
    def freeze_model(model, device):
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

    def train(
            self,
            dataloader,
            x_init: torch.Tensor,
            y_init: torch.Tensor,

            x_val: torch.Tensor,
            y_val: torch.Tensor,

            n_steps=1e4,
            val_log_frequency=10
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
            :param dataloader: DataLoader containing the training data.
            :param x_init: Initial context points (T x B x dim).
            :param y_init: Initial context labels (T x B).
            :param n_steps: Number of distillation steps.
            :param val_log_frequency: Frequency of validation logging.
        """
        # map the fidelity and hp [0,1] to the logit space (which we optimize)

        x = deepcopy(x_init)
        y = deepcopy(y_init)

        p = x_init[:, :, 1:]
        # to enforce the [0,1] constraint, we first map the values that actually
        # live in the [0,1] space to the logit space (because during the loop we will apply sigmoid)
        x_init[:, :, 1:] = torch.log(p / (1 - p + 1e-6))

        distilled_x_latent = nn.Parameter(x_init)
        distilled_y = nn.Parameter(y_init)

        distilled_x_latent = distilled_x_latent.to(self.device)
        distilled_y = distilled_y.to(self.device)

        self.optimizer = self.optimizer_partial([distilled_x_latent, distilled_y])

        self.pbar = tqdm(range(int(n_steps)), desc="Distillation Progress", unit="step")
        for step in self.pbar:
            losses = []
            for x_query, y_query in dataloader:
                x_query = x_query.to(self.device)  # T x B x dim
                y_query = y_query.to(self.device)  # T x B

                # fixme: should we optimize for hold out data instead?
                loss = self.train_step(
                    x_query, y_query,
                    distilled_points=constrain(distilled_x_latent, temp=1),
                    distilled_labels=distilled_y
                )
                losses.append(loss)

                # log.debug(
                #     f"Step {step}: Loss: {loss.item()}, Distilled Points: {constrain(distilled_x_latent).squeeze(1)}")

            self.logger.add_scalar(
                "train_loss",
                torch.mean(torch.stack(losses)).item(),
                distilled_x_latent.shape[0], step
            )

            if step % val_log_frequency == 0:
                # testing on the same task, how well we predict the GT data given the distillation
                x_query, y_query = dataloader.dataset.get_shuffled_data()

                reconstrucion_loss = self.test_on_new_task(
                    model=self.model,
                    # fixme: temp
                    prefix_x=constrain(distilled_x_latent.detach(), temp=1),
                    prefix_y=distilled_y,
                    context_task_x=x_query,
                    context_task_y=y_query,
                    # fixme: will this be available during inference? -- no
                    query_task_x=x_val,
                    query_task_y=y_val,
                    step=step,
                    task_name="nll_distilled_reconstruction"
                )

                self.pbar.set_postfix(
                    val_loss=reconstrucion_loss[0],
                    train_loss=torch.mean(torch.stack(losses)).item()
                )

        # distilled_x_latent = distilled_x_latent.detach().cpu()
        # distilled_x_latent[:, :, 1:] = torch.sigmoid(distilled_x_latent[:, :, 1:]).detach().cpu()
        distilled_x = constrain(distilled_x_latent, temp=1).detach().cpu()

        if torch.allclose(x, distilled_x) or torch.allclose(y, distilled_y):
            warnings.warn(
                "Distillation did not change the points. Check your optimizer and learning rate."
            )
        return distilled_x, distilled_y

    def train_step(self, x_query, y_query, distilled_points, distilled_labels):
        T, B, dim = x_query.shape

        # ENFORCE PARAMAMETER CONSTRAINTS ----------------------------------
        # The transformermodel places certain constraints on the fwd.

        # to ensure numerical stability:
        # with torch.no_grad():
        #     constrained_tensor = torch.sigmoid(constrained_tensor).clamp(1e-3, 1 - 1e-3)

        # Get model predictions using distilled context
        logits = self.model(
            # ([x_train, x_query], ytrain)
            (  # notice, that x_query comes from the dataset and thus is constrained
                torch.cat([distilled_points.expand(-1, B, -1), x_query], dim=0),
                distilled_labels.expand(-1, B)
            ),
            single_eval_pos=distilled_points.shape[0]
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
    import tempfile

    BATCH_SIZE = 64
    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 50  # FIXME: important hyperparameter
    N_STEPS = 10  # number of distillation steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=2, single_eval_pos=500)

    sep = batch.single_eval_pos[0]
    x, y = batch.x, batch.y

    # 0 task is the one we want to predict on
    target_task_context_x = x[:sep, 0:1]
    target_task_context_y = y[:sep, 0:1]
    target_task_val_x = x[sep:, 0:1]
    target_task_val_y = y[sep:, 0:1]

    # fixme: max context size is 1k, so distilled + task context + query can
    #  exceed that. in that case we will want to batch over the exceeing query points with the
    #  same context.
    max_query_size = 1000 - target_task_context_x.shape[0] - DISTILL_SIZE
    target_task_query_x = x[sep:sep + max_query_size, 0:1]  # actual test points
    target_task_query_y = y[sep:sep + max_query_size, 0:1]  # actual test labels

    x_to_distill = x[:sep, 1:]
    y_to_distill = y[:sep, 1:]
    x_val = x[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    y_val = y[sep:sep + max_query_size, 0:1]  # fixme: max_query_size
    print(x.shape, y.shape)

    # communicate it to the dataset.
    dtrain = DTrain(x_to_distill, y_to_distill, length=BATCH_SIZE, sequence_length_max=500)
    # x_query, y_query = dtrain[0]

    # get random subset of the data for initialization of the distillation
    x_init, y_init = dtrain.get_rnd_subset_initialization(DISTILL_SIZE)

    dataloader = torch.utils.data.DataLoader(
        dtrain,
        collate_fn=partial(collate, min_length=10, max_length=500),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger = BufferedFileLogger(
            file_name="distill.csv",
            file_path=tmpdirname,
            buffer_size=1000,
            header=("metric", "value", "context_size", "global_step"))

        trainer = DistillContextTrainer(
            model=ftpfn,
            optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4),
            criterion=pfn_backend.criterion,
            logger=logger,
            device=device
        )

        distilled_x, distilled_y = trainer.train(
            dataloader=dataloader,
            x_init=x_init,
            y_init=y_init,
            x_val=x_val,
            y_val=y_val,
            n_steps=N_STEPS,
            val_log_frequency=1
        )

        # Plot distilled reconstruction loss. --------------------------------
        import matplotlib.pyplot as plt

        ax = logger.plot_scalar_curve(
            metric="nll_distilled_reconstruction",
            title="Distilled Reconstruction Loss",
            x='global_step',
            plot=False
        )
        ax = logger.plot_scalar_curve(
            metric="train_loss",
            title="Distilled Training Loss",
            x='global_step',
            plot=False,
            ax=ax
        )
        ax.set_xlabel("Context size")
        ax.set_ylabel("NLL Loss")
        ax.set_title("Distilled Reconstruction Loss")
        plt.show()

        # Plot downstream task performance with distilled context. ---------------
        #FIXME: move this into a separate evaluation function based of the evaluator class
        CONTEXT_SIZES = range(10, target_task_context_x.shape[0], 20)
        losses = trainer.test_on_new_task(
            model=pfn_backend,
            task_name='incl. distilled context from task 0',
            prefix_x=distilled_x,
            prefix_y=distilled_y,
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # Sanity check: the initial points of optimization added as context
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='x_init context (no-distillation)',
            prefix_x=x_init,
            prefix_y=y_init,
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # No distillation: Should be what the ifbo paper reports
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (no distillation)',
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # adding in half of the related dataset in
        half_x = x_to_distill.shape[0] // 2
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (half context task 0)',
            prefix_x=x_to_distill[:half_x],
            prefix_y=y_to_distill[:half_x],
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES
        )

        # Adding in (almost) the entire context of the related task
        # we can't fit the entire one, since we need space in the sequence
        # to do a batched evaluation over the query points
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (approx. complete context task 0)',
            prefix_x=x_to_distill,
            prefix_y=y_to_distill,
            context_task_x=target_task_context_x[:-25],
            context_task_y=target_task_context_y[:-25],
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=CONTEXT_SIZES  # [:10]
        )

        ax = None
        plots = [
            'incl. distilled context from task 0',
            'x_init context (no-distillation)',
            'baseline (no distillation)',
            'baseline (approx. complete context task 0)',
            'baseline (half context task 0)'
        ]
        for metric in plots:
            try:
                ax = logger.plot_scalar_curve(
                    metric=metric,
                    plot=False,
                    ax=ax,
                    x='context_size',
                    y='value',
                )
            except ValueError:
                log.warning(f"Metric {metric} not found in logger.")
                continue

        ax.set_xlabel("Task's Context size")
        ax.set_ylabel("NLL Loss")
        ax.set_title("Distilled Context + increments of the new Task (nll)", )
        ax.legend()
        plt.show()
