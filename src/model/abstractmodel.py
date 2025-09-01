import logging
from typing import Union

import torch

from ifbo.transformer import TransformerModel
from src.model.components.contender_bonus import BudgetBasedPIBonus
from utils.dotdict import DotDict

log = logging.getLogger(__name__)


class IFBOInterface:

    def pre_train(self, tasks):
        pass

    def train(self, *args, **kwargs):
        """
        returns the prefix (x,y) for the new task; necessary for the test_on_task to comply
        with the api in case of distillation
        """
        return None

    def forward(self, *args, **kwargs):
        """
        Interface for the TransformerModel class from ifbo.
        """
        if isinstance(args, tuple) and len(args) == 1:
            # this is the unfortunate TransformerModel compatability
            args = args[0]
            x, y = args
            single_eval_pos = kwargs['single_eval_pos']

            mask = x[:, 0] >= 1000
            x[mask, 0] = torch.tensor(999.)

            kwargs = {
                'context_x': x[:single_eval_pos],
                'context_y': y,
                'query_x': x[single_eval_pos:]
            }
        elif {'x_train', 'y_train', 'x_test'}.issubset(set(kwargs.keys())):
            # during ifbo deployment
            # curve id encoder will struggle with index positions longer then the
            # sequence length --> so we just replace them here. They don't have any meaning anyways!
            mask = kwargs['x_train'][:, 0] >= 1000
            kwargs['x_train'][mask, 0] = torch.tensor(999.)

            mask = kwargs['x_test'][:, 0] >= 1000
            kwargs['x_test'][mask, 0] = torch.tensor(999.)

            kwargs = {
                'context_x': kwargs['x_train'].unsqueeze(1),
                'context_y': kwargs['y_train'].unsqueeze(1),
                'query_x': kwargs['x_test'].unsqueeze(1)
            }

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            kwargs = {k: v.to(device) for k, v in kwargs.items()}

        if 'single_eval_pos' in kwargs:
            del kwargs['single_eval_pos']

        return self._forward(**kwargs)

    def _forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def get_ei(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        out = self.smbo(
            x_train=x_train, y_train=y_train, x_test=x_test, inc=inc, minimize=minimize
        )

        if out['predictions'] is not None:
            return self.criterion.ei(
                out['predictions'].squeeze(1),
                best_f=inc,
                maximize=True
            )
        else:
            return out['acq_values']

    @torch.no_grad()
    def get_pi(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        out = self.smbo(
            x_train=x_train, y_train=y_train, x_test=x_test, inc=inc,
            minimize=minimize
        )

        if out['predictions'] is not None:
            return self.criterion.pi(
                out['predictions'].squeeze(1),
                best_f=inc,
                maximize=True
            )
        else:
            return out['acq_values']


class AbstractModel(IFBOInterface):
    __name__ = "AbstractModel"

    def __init__(
            self,
            initial_design,
            strategy,
            flippable,
            logger,
            model,
            device,
            related_task_data,
            weights=None,  # weight strategy
            imputer=None,
            callbacks=(),
            contender_bonus: Union[None, BudgetBasedPIBonus] = None
            # **kwargs
    ):
        """
        Abstract base class for models that can be trained and evaluated on tasks.

        :param flippable: Whether the model can flip the labels.
        :param logger: Logger for logging information.
        :param criterion: Loss function or evaluation metric.
        :param model: The underlying model to be used.
        :param device: Device to run the model on (e.g., 'cpu' or 'cuda').
        :param related_task_data: Data from related tasks for context.
        :param callbacks: List of callbacks to be executed during training/evaluation.
        :param kwargs: Additional keyword arguments for customization.
        """
        self.interim_results = {}
        self.initial_design = initial_design
        self.strategy = strategy
        self.imputer = imputer
        self.weights = weights  # weight strategy,
        self.flippable_related = flippable
        self.logger = logger

        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model
        self.model.eval()
        self.criterion = self.model.criterion

        self.device = device
        self.related_task_data = related_task_data
        self.callbacks = callbacks if callbacks is not None else []
        self.contender_bonus = contender_bonus
        # self.kwargs = kwargs

        self.__name__ = f'{self.__class__.__name__}_{self.strategy.__name__}'

        self.initialized = False  # initializing the related_context during first call to meet
        # flipping needs
        self.num_related = None
        self.related_context = None

        self.initialized = False
        self.related_context = None  # Will be initialized during the first call to forward
        self.num_related = None  # Number of related tasks, initialized during the first call to forward
        self.callback_kwargs = []

    def __post_init__(self, related_context_data, minimize):
        related_context_x = related_context_data.x
        related_context_y = related_context_data.y
        padding_mask = related_context_data.padding_mask

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)

        related_context_x = related_context_x.to(self.device)
        related_context_y = related_context_y.to(self.device)

        # related_context_y=related_context_y * 1.1 - 0.2
        # min_val = transformed.min()
        # max_val = transformed.max()
        # related_context_y = (transformed - min_val) / (max_val - min_val)

        self.related_context = DotDict({
            'x': related_context_x,
            'y': related_context_y,
            'padding_mask': padding_mask
        })

        self.num_related = self.related_context.x.shape[1]

        cbs = []
        for callback in self.callbacks:
            cbs.append(
                callback(
                    parent_model=self,
                    model=self.model,
                    related_context=self.related_context,
                    logger=self.logger,
                    device=self.device,
                    # **self.kwargs,
                    # pass additional kwargs if needed

                )
            )
        self.callbacks = cbs

        # todo harmonize the post_init!
        self.initial_design.__post_init__(
            parent_model=self,
            model=self.model,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device,
            callbacks=self.callbacks,
        )

        self.strategy.__post_init__(
            parent_model=self,
            model=self.model,
            criterion=self.criterion,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device,
            callbacks=self.callbacks,
        )

        if self.imputer is not None:
            self.imputer.__post_init__(
                parent_model=self,
                model=self.model,
                criterion=self.criterion,
                related_context=self.related_context,
                logger=self.logger,
                device=self.device
            )

        if self.weights is not None:
            self.weights.__post_init__(
                parent_model=self,
                model=self.model,
                related_context=self.related_context,
                logger=self.logger,
                device=self.device
            )

        self.initialized = True

    def _preprocess(self, x_train, y_train, x_test, inc, minimize=True):

        if not self.initialized:
            self.__post_init__(self.related_task_data, minimize)

        y_train = y_train.to(self.device)
        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        inc = inc.to(self.device)

        y_train = y_train.unsqueeze(1)
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

        if torch.any(x_test > 999.):
            log.warning(f"Query points x_test contain values > 999: {x_test[x_test > 999.]}")
            x_test = torch.clamp(x_test, max=999.)
        if torch.any(x_train > 999.):
            log.warning(f"Training points x_train contain values > 999: {x_train[x_train > 999.]}")
            x_train = torch.clamp(x_train, max=999.)

            # (Adjust searchspaces) ------------------------------------------------
        # in case the search spaces are supersets of each other, we need to augment the
        # x_train data to match the related context (we need to drop the dim later for the
        # acquisition function to not notice)
        if x_train.shape[-1] < self.related_context.x.shape[-1]:
            diff = self.related_context.x.shape[-1] - x_train.shape[-1]
            placeholder = self.related_context.x[:, :, -diff:].mean(dim=1).mean(dim=0)
            x_train = torch.cat([
                x_train,
                placeholder.repeat(x_train.shape[0], 1).unsqueeze(1)
            ], dim=-1).to(self.device)

            x_test = torch.cat([
                x_test,
                placeholder.repeat(x_test.shape[0], 1).unsqueeze(1)
            ], dim=-1).to(self.device)

        return x_train, y_train, x_test, inc

    def smbo(self, x_train, y_train, x_test, inc, minimize=True):
        step = x_train.shape[0]
        x_train, y_train, x_test, inc, = \
            self._preprocess(x_train, y_train, x_test, inc, minimize=minimize)

        for callback in self.callbacks:
            callback.on_acq_start(x_train, y_train, x_test, inc)

        # (Impute related tasks) -----------------------------------------------
        if self.imputer is not None:
            imputed_y = self.imputer(
                x_train=self.related_context.x,
                x_test=x_train.repeat(1, self.num_related, 1),
                y_train=self.related_context.y
            )

            self.interim_results.update(dict(imputed_y=imputed_y, step=step))

        if step < self.initial_design.size:
            pi_values = self.initial_design(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                inc=inc,
                acquisition_fn='pi'
            )

            for callback in self.callbacks:
                callback.on_acq_end_warmstart(x_train, y_train, x_test, inc, pi_values)

            acq = pi_values
            predictions = None


        else:
            acq = None
            predictions = self.strategy(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                inc=inc,
            )
            for callback in self.callbacks:
                callback.on_acq_end_mixture(x_train, y_train, x_test, inc, predictions)

        if self.contender_bonus is not None:
            # Encouraging in-depth exploration of configurations that have already been evaluated
            # by adding a small bonus to their acquisition values.
            # This reflects the belief, that a budget token to advance an existing configuration
            # usually is informationally more valuable than a token to start a new configuration.
            if predictions is not None:
                acq = self.criterion.pi(
                    predictions.squeeze(1),
                    best_f=inc,
                    maximize=True
                )

            acq = self.contender_bonus(
                x_train, y_train, x_test,
                acq, inc
            )

            # min_fid = torch.min(x_test).item()
            # is_contender = (x_test[:,0,1] != min_fid)
            # acq[is_contender] += self.contender_bonus
            # acq = torch.clamp(acq, min=0, max=1)

            if False:
                import matplotlib.pyplot as plt

                plt.hist(acq[is_contender], color='orange', label='x_train', alpha=0.1,
                         density=True)
                plt.hist(acq[~is_contender], color='blue', label='x_test', alpha=0.5,
                         density=True)

                plt.legend()
                plt.show()

        # we just need the last one for the next round
        self.interim_results['last-x_train'] = x_train

        return {"acq_values": acq, "predictions": predictions}
