import warnings
from copy import deepcopy
from functools import partial
from typing import Union, Tuple

import torch
from ifbo import BarDistribution, FTPFN
from ifbo.transformer import TransformerModel
from torch import nn
from tqdm import tqdm

from src.dataset.dataloading import DTrain, collate
from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.model.abstractmodel import AbstractModel
from src.model.batch_padded_pfn import parse_batch_for_padded_train_data
from src.utils.filelogger import BufferedFileLogger

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


class DistillContext(AbstractModel):
    def __init__(self,
                 model: Union[FTPFN, TransformerModel],
                 optimizer: partial,
                 logger: BufferedFileLogger,
                 related_task_data,
                 criterion: BarDistribution = None,
                 device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                 temperature_annealing: AdaptiveTemperature = None,
                 distillsize=100
                 ):
        """

         Args:
        :param dataloader: DataLoader containing the training data.
        :param model: The model to be distilled.
        :param criterion:
        :param logger:
        """

        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model

        self.device = device
        self.freeze_model(self.model, device)

        self.distillsize = distillsize

        self.optimizer_partial = optimizer
        self.optimizer = None
        self.criterion = criterion if criterion is not None else model.criterion
        self.related_task_data = related_task_data

        self.logger = logger
        self.pbar = None

        self.evaluator = TestOnNewTaskNLL(self.criterion, self.logger, self.device)
        self.test_on_new_task = self.evaluator.test_on_new_task

        if temperature_annealing is None:
            self.temperature_annealing = AdaptiveTemperature()
        else:
            self.temperature = 1

        # dummy initializaiton
        self.distilled_x, distilled_y = torch.tensor([]), torch.tensor([])

    @staticmethod
    def freeze_model(model, device):
        model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

    def sample_initial_distillation(
            self, x: torch.Tensor, y: torch.Tensor, padding_mask=None,
            size: int = 100
    ):
        """
        Sample a random subset of the data for initialization of the distillation.
        :param x: Input data tensor.
        :param y: Labels tensor.
        :param size: Size of the subset to sample.
        :return: Tuple of sampled input data and labels.
        """
        T, B, D = x.shape

        if padding_mask is None:
            padding_mask = torch.zeros((T, B), dtype=torch.bool)

        # Store the sampled indices for each batch element
        sampled_indices = []
        for b in range(B):
            valid_positions = torch.where(torch.logical_not(padding_mask[b, :]))[0]
            if len(valid_positions) < size:
                raise ValueError(
                    f"Not enough valid positions in batch {b} to sample {size} times.")
            chosen = valid_positions[torch.randperm(len(valid_positions))[:size]]
            sampled_indices.append(chosen)

        returned_x = torch.stack([x[chosen, b, :] for b, chosen in enumerate(sampled_indices)],
                                 dim=1)
        returned_y = torch.stack([y[chosen, b] for b, chosen in enumerate(sampled_indices)], dim=1)

        # Ensure the returned tensors have the same shape as the input
        assert returned_x.shape == (size, B,
                                    D), f"Expected shape {(size, B, D)}, got {returned_x.shape}"
        assert returned_y.shape == (size, B), f"Expected shape {(size, B)}, got {returned_y.shape}"

        return returned_x, returned_y

    def train(
            self,
            n_steps=1e4,
            val_log_frequency=10,
            batch_size=64,
            temperature=1,
            query_task_x=None,  # used for validation only
            query_task_y=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Ma et al 2024 In Context Data Distillation with TabPFN:
    
        Given a new dataset D_train and a query point x to classify, its class probability is
        computed as p θ(y|x, D_train), where D_train can be understood as the “prompt”. In-Context
        Distillation (ICD) – similarly to prompt-tuning (Lester et al., 2021)
        – optimizes the likelihood of the real data given the distilled one:
        L(D_train --> D_dist) = - E_{(x,y)~D_train)} [log p(y|x, D_dist)]
        Then, given a new point x to classify, we use D_dist as the context and predict the label
        arg max y p θ(y|x, D_dist)
    
        Args:
            :param n_steps: Number of distillation steps.
            :param val_log_frequency: Frequency of validation logging.
        """
        # map the fidelity and hp [0,1] to the logit space (which we optimize)

        x, y = self.related_task_data.x, self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask

        # (0) Get the initial distillation points
        distilled_x, distilled_y = self.sample_initial_distillation(
            x, y,
            padding_mask=padding_mask,
            size=self.distillsize
        )

        self._init_x = deepcopy(distilled_x)
        self._init_y = deepcopy(distilled_y)

        # # to enforce the [0,1] constraint, we first map the values that actually
        # # live in the [0,1] space to the logit space (because during the loop we will apply sigmoid)
        p = distilled_x[:, :, 1:]
        distilled_x_latent = distilled_x.clone()
        distilled_x_latent[:, :, 1:] = torch.log(p / (1 - p + 1e-6))

        distilled_x_latent = nn.Parameter(distilled_x_latent)
        distilled_y = nn.Parameter(distilled_y)

        self.optimizer = self.optimizer_partial([distilled_x_latent, distilled_y])
        distilled_x_latent = distilled_x_latent.to(self.device)
        distilled_y = distilled_y.to(self.device)

        # (1) set up the dataloader
        dataset = DTrain(
            x, y, padding_mask,
            # sequence_length_max=self.related_task_data.single_eval_pos,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate, min_length=10),
        )

        self.pbar = tqdm(range(int(n_steps)), desc="Distillation Progress", unit="step")
        for step in self.pbar:
            losses = []
            for x_query, y_query, padding_mask in dataloader:
                # tile the query points
                # padding_mask.shape is (batch, n_tasks, T)
                batch, T, n_tasks, dim = x_query.shape
                tiled_batch = batch * n_tasks

                x_query = x_query.to(self.device)
                y_query = y_query.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # dummy
                # x_query = torch.cat([
                #     torch.zeros(batch, T, 1, dim),
                #     torch.ones(batch, T, 1, dim)
                # ], dim=2)  # Now shape: (batch, T, 2, dim)
                # padding_mask = torch.cat([
                #     torch.zeros(batch, 1, T),
                #     torch.ones(batch, 1, T)
                # ], dim=1)

                x_query = x_query.permute(1, 0, 2, 3).reshape(T, tiled_batch,dim)  # (T, batch * 2, dim)
                y_query = y_query.permute(1, 0, 2).reshape(T, tiled_batch)
                padding_mask_tiled = padding_mask.reshape(tiled_batch, T)


                x_query = x_query.to(self.device)
                y_query = y_query.to(self.device)

                # dummy
                # distilled_x_latent = torch.cat([
                #     torch.zeros(T, 1, dim),
                #     torch.ones(T, 1, dim)
                # ], dim=1)  # (T, 2, dim)

                # Repeat distilled_x n_tasks times across batch dimension
                distilled_x_latent_tiled = distilled_x_latent.repeat(1, batch, 1)
                # (T,
                # batch*2, dim)
                distilled_y_tiled = distilled_y.repeat(1, batch)  # (T, batch*2)
                # on the dummy
                # assert distilled_x_latent_tiled.shape == (T, tiled_batch, dim)
                # assert x_query.shape == (T, tiled_batch, dim)
                # assert distilled_x_latent_tiled.shape == (T, tiled_batch, dim)
                # assert (x_query + distilled_x_latent_tiled).unique().tolist() == [0, 2]
                # assert torch.all(x_query[: , 0, :] == 0) and torch.all(distilled_x_latent_tiled[: , 0, :] == 0)
                # assert torch.all(x_query[: , 1, :] == 1) and torch.all(distilled_x_latent_tiled[: , 1, :] == 1)
                # assert torch.all(padding_mask_tiled[0] == 0)
                # assert torch.all(padding_mask_tiled[1] == 1) # this one fails

                # since we will cat([distilled_x, x_query], dim=0) in the model
                # the padding needs to be extended:
                # padding_mask_tiled = torch.cat([
                #     torch.zeros((tiled_batch, distilled_x_latent.shape[0]), device=self.device,
                #                 dtype=torch.bool),
                #     padding_mask_tiled,
                #
                # ], dim=1)

                loss = self.train_step(
                    x_query, y_query,
                    # constrain the learnable parameters:
                    distilled_x=constrain(distilled_x_latent_tiled, temp=temperature),
                    distilled_y=distilled_y_tiled,
                    padding_mask=padding_mask_tiled,
                    batch=batch_size,
                    n_tasks=n_tasks
                )
                losses.append(loss)

                # log.debug(
                #     f"Step {step}: Loss: {loss.item()}, Distilled Points: {constrain(distilled_x_latent).squeeze(1)}")
            train_losses = torch.mean(torch.stack(losses, dim=0), dim=0)
            for i, loss in enumerate(train_losses):
                self.logger.add_scalar(
                    f"train_loss",
                    loss.item(),
                    step,
                    distilled_x_latent.shape[0],
                    i
                )


            if step % val_log_frequency == 0:
                reconstruction, predictive = self.validation_step(
                    query_task_x=query_task_x,
                    query_task_y=query_task_y,
                    distilled_x_latent=distilled_x_latent,
                    distilled_y=distilled_y,
                    temperature=temperature,
                    step=step,
                    dataset=dataset
                )

                self.pbar.set_postfix(
                    val_loss=reconstruction[0],
                    train_loss=train_losses,
                    predictive=predictive[0],
                )

        # distilled_x_latent = distilled_x_latent.detach().cpu()
        # distilled_x_latent[:, :, 1:] = torch.sigmoid(distilled_x_latent[:, :, 1:]).detach().cpu()
        distilled_x = constrain(distilled_x_latent, temp=1).detach().cpu()

        if torch.allclose(self._init_x, distilled_x) or torch.allclose(self._init_y, distilled_y):
            warnings.warn(
                "Distillation did not change the points. Check your optimizer and learning rate."
            )

        self.distilled_x = distilled_x
        self.distilled_y = distilled_y
        return distilled_x, distilled_y

    def validation_step(self, query_task_x, query_task_y, distilled_x_latent, distilled_y,
                        temperature, step, dataset):
        # testing on the same task, how well we predict the GT data given the distillation
        x_context, y_context, padding = dataset[0]
        # x_context = x_context.permute(1, 0, 2).to(self.device)
        # y_context = y_context.permute(1, 0).to(self.device)

        # check how well we can reconstruct the data given the distillation
        reconstruction_loss = self.test_on_new_task(
            model=self.model,
            # fixme: temp
            prefix_x=constrain(distilled_x_latent.detach(), temp=temperature),
            prefix_y=distilled_y,
            context_task_x=torch.tensor([], device=self.device),
            context_task_y=torch.tensor([], device=self.device),
            # fixme: will this be available during inference? -- no
            query_task_x=x_context,
            query_task_y=y_context,
            step=step,
            task_name="nll_distilled_reconstruction"
        )

        if query_task_x is not None and query_task_y is not None:
            # generalization capability to the task of interest
            predictive_loss = self.test_on_new_task(
                model=self.model,
                # fixme: temp
                prefix_x=constrain(distilled_x_latent.detach(), temp=temperature),
                prefix_y=distilled_y,
                context_task_x=torch.tensor([], device=self.device),
                context_task_y=torch.tensor([], device=self.device),
                # fixme: will this be available during inference? -- no
                query_task_x=query_task_x.repeat(1, x_context.shape[1], 1),
                query_task_y=query_task_y.repeat(1, x_context.shape[1]),
                step=step,
                task_name="nll_distilled_predictive"
            )

        return reconstruction_loss, predictive_loss

    def train_step(self, x_query, y_query, padding_mask, distilled_x, distilled_y, batch, n_tasks):

        # to ensure numerical stability:
        # with torch.no_grad():
        #     constrained_tensor = torch.sigmoid(constrained_tensor).clamp(1e-3, 1 - 1e-3)

        # Get model predictions using distilled context
        logits = self.model(
            # ([x_train, x_query], ytrain)
            (  # notice, that x_query comes from the dataset and thus is constrained
                torch.cat([distilled_x, x_query], dim=0),
                distilled_y
            ),
            single_eval_pos=distilled_x.shape[0],
            # src_key_padding_mask = padding_mask
        )

        # Compute the BarDistribution NLL loss
        loss = self.criterion(logits, y_query)  # y_query is already tiled
        # Found in PFNs4HPO.pfns4hpo.train.py: sometimes the seq length can be one off
        # that is because bar dist appends the mean
        loss = loss.view(-1, logits.shape[1])

        # Step 1: Convert padding_mask to "valid mask", shape (T, B)
        valid_mask = ~padding_mask

        # Step 2: Mask the loss
        masked_loss = loss.T * valid_mask

        # Step 3: Count valid tokens per batch
        valid_counts = valid_mask.sum(dim=1)  # (64,) — number of valid tokens per batch

        # Step 4: Sum losses per batch and divide by valid count
        loss_sum_per_batch = masked_loss.sum(dim=1)  # (64,)
        loss_mean_per_batch = loss_sum_per_batch / valid_counts.clamp(min=1)

        losses_per_task = loss_mean_per_batch.view(int(loss.shape[1] / n_tasks),n_tasks ).mean(
            dim=0)  #
        # (64, 2)

        train_loss = losses_per_task.sum()

        # todo undo the tiling for loss and report the mean over the task dim
        # task_losses = torch.mean(loss, dim=0)  # related task wise losses

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        return losses_per_task

    def forward(self,xy, single_eval_pos=0, src_key_padding_mask=None,):
        """
        Performs the forward pass of the model, taking query points, query labels, distilled
        points, and distilled labels as inputs. The method processes the input data by
        concatenating distilled points and query points, alongside expanding and formatting
        the corresponding labels. The combined data is then forwarded through the model
        to produce logits. The logits provide the necessary output for further computations
        such as loss calculation or evaluation.

        :param x_query: Query points for the forward pass. The tensor is expected to have
            dimensions (T, B, dim), where T is the sequence length, B is the batch size,
            and dim is the feature dimension.
        :param y_query: Labels corresponding to the query points. Used in the context for
            further computations involving the model output.
        :param distilled_points: Pre-distilled points created during training or refinement
            stages. These are used as part of the input to the model.
        :param distilled_labels: Labels corresponding to the distilled points. These are
            expanded during the forward pass to match the batch size.
        :return: The computed logits after passing the processed inputs through the model.
        """
        x, y = xy

        T, B, dim = x.shape
        x_context = x[:single_eval_pos, :, :]
        y_context = y[:single_eval_pos, :]
        x_query = x[single_eval_pos:, :, :]

        logits = self.model(
            # ([x_train, x_query], ytrain)
            (  # notice, that x_query comes from the dataset and thus is constrained
                torch.cat([self.distilled_x.expand(-1, B, -1), x_context, x_query], dim=0),
                torch.cat([self.distilled_y.expand(-1, B), y_context], dim=0)
            ),
            single_eval_pos=self.distilled_x.shape[0] + x_context.shape[0]
        )

        return logits

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    import ifbo

    from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch
    from src.dataset.dataloading import collate
    import tempfile

    N_TASKS=4
    BATCH_SIZE = 16
    # TODO initialize distilled points with subset of x with appropriate size
    DISTILL_SIZE = 50  # FIXME: important hyperparameter
    N_STEPS = 200  # number of distillation steps

    device = torch.device( "cpu")
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=N_TASKS, single_eval_pos=[500, 200, 300, 400], alphas=None)

    padded_batch = parse_batch_for_padded_train_data(batch, target_idx=0)

    related_task_data = padded_batch.related_tasks
    task_data = padded_batch.target_task

    target_task_context_x = task_data.x
    target_task_context_y = task_data.y
    target_task_query_x = task_data.query_x
    target_task_query_y = task_data.query_y
    padding_mask = task_data.padding_mask


    with tempfile.TemporaryDirectory() as tmpdirname:
        logger = BufferedFileLogger(
            file_name="distill.csv",
            file_path=tmpdirname,
            buffer_size=1000,
            header=("metric", "value", "context_size", "global_step", "related_task_id"))

        trainer = DistillContext(
            model=ftpfn,
            optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4),
            criterion=pfn_backend.criterion,
            related_task_data=related_task_data,
            logger=logger,
            device=device,
            distillsize=DISTILL_SIZE,
        )

        distilled_x, distilled_y = trainer.train(
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
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
        # FIXME: move this into a separate evaluation function based of the evaluator class
        CONTEXT_SIZES = range(10, target_task_context_x.shape[0], 20)
        n_related_tasks = related_task_data.x.shape[1]
        losses = trainer.test_on_new_task(
            model=pfn_backend,
            task_name='incl. distilled context',
            prefix_x=distilled_x,
            prefix_y=distilled_y,
            context_task_x=target_task_context_x.repeat(1, n_related_tasks, 1),
            context_task_y=target_task_context_y.repeat(1, n_related_tasks),
            query_task_x=target_task_query_x.repeat(1, n_related_tasks, 1),
            query_task_y=target_task_query_y.repeat(1, n_related_tasks),
            context_sizes=CONTEXT_SIZES
        )

        # Sanity check: the initial points of optimization added as context
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='x_init context (no-distillation)',
            prefix_x=trainer._init_x,
            prefix_y=trainer._init_y,
            context_task_x=target_task_context_x.repeat(1, n_related_tasks, 1),
            context_task_y=target_task_context_y.repeat(1, n_related_tasks),
            query_task_x=target_task_query_x.repeat(1, n_related_tasks, 1),
            query_task_y=target_task_query_y.repeat(1, n_related_tasks),
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
        x_to_distill = related_task_data.x
        y_to_distill = related_task_data.y
        padding_mask = related_task_data.padding_mask
        half_x = x_to_distill.shape[0] // 2
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (half context related task)',
            prefix_x=x_to_distill[:half_x],
            prefix_y=y_to_distill[:half_x],
            context_task_x=target_task_context_x.repeat(1, n_related_tasks,1),
            context_task_y=target_task_context_y.repeat(1, n_related_tasks),
            query_task_x=target_task_query_x.repeat(1, n_related_tasks,1),
            query_task_y=target_task_query_y.repeat(1, n_related_tasks),
            context_sizes=CONTEXT_SIZES,
            src_key_padding_mask=padding_mask[:half_x]
        )

        # Adding in (almost) the entire context of the related task
        # we can't fit the entire one, since we need space in the sequence
        # to do a batched evaluation over the query points
        trainer.test_on_new_task(
            model=pfn_backend,
            task_name='baseline (approx. complete context related task)',
            prefix_x=x_to_distill,
            prefix_y=y_to_distill,
            context_task_x=target_task_context_x[:-25].repeat(1, n_related_tasks,1),
            context_task_y=target_task_context_y[:-25].repeat(1, n_related_tasks),
            query_task_x=target_task_query_x.repeat(1, n_related_tasks,1),
            query_task_y=target_task_query_y.repeat(1, n_related_tasks),
            context_sizes=CONTEXT_SIZES,  # [:10]
            src_key_padding_mask=padding_mask
        )

        ax = None
        plots = [
            'incl. distilled context',
            'x_init context (no-distillation)',
            'baseline (no distillation)',
            'baseline (approx. complete context related task)',
            'baseline (half context related task)'
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
