import pfns4bo
import torch
import warnings
import logging

from model.components.calc_reliability import calc_target_cv_nll
from model.components.ema_filter import ema_conv_causal
from model.components.error_model_wrapper import WrappedErrorModel
from model.probability_conv.convolver import DistributionConvolver
from model.probability_conv.map_binnings import project_probs_to_new_grid

log = logging.getLogger(__name__)


class AbstractWeights:
    """
    Abstract class for computing weights.
    """

    def __post_init__(self, related_context, device, logger, model, parent_model=None):
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.device = device
        self.logger = logger
        self.parent_model = parent_model
        self.model = model

    def __call__(self, x_train, *args, **kwargs) -> torch.Tensor:
        """
        Compute the weights.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class MeanWeights(AbstractWeights):
    """
    Class to compute the mean of a list of weights.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the mean of the weights.
        """
        n = self.num_related + 1
        weights = torch.ones(n, device=self.device) / n
        return weights


class PriorWeights(AbstractWeights):
    """
    Class to compute the prior weights.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the prior weights.
        """
        n = self.num_related + 1
        weights = torch.ones(n, device=self.device)
        weights[0] = 0.0  # Set the first weight to zero
        weights[1:] = 1 / (n - 1)  #
        return weights


def override_call_decorator(func):
    def wrapper(x_train, x_test, *args, **kwargs):
        n_x_test = x_test.shape[0]

        # we must consider, that we ask for the next fidelity of seen configurations
        train = x_train.clone()
        min_fidelity = train[:, :, 1].min()
        train[:, :, 1] += min_fidelity

        # we need to also consider, that in x_test we ask unseen configurations
        test = x_test.clone()
        test[:, :, 1] = min_fidelity

        # METHOD OVERRIDE!
        result_logits = func(
            x_train=x_train,
            x_test=torch.cat([x_test, train, test], dim=0) ,
                   *args, **kwargs
        )
        test_logits = result_logits[:n_x_test, :, :]
        lookahead_logits = result_logits[n_x_test:, :, :]

        # gain access to the parent model
        self = getattr(func, '__self__', None)
        self.parent_model.interim_results.update({
            f'last-{func.__name__}-lookahead' : lookahead_logits,
            f'x_lookahead': torch.cat([train[:, 0:1, :], test[:, 0:1, :]],dim=0)

        })

        return test_logits
    return wrapper

class PastSurpriseWeights(AbstractWeights):

    def __init__(
            self,
            alpha=0.1,
            bias_correction=True,
            truncate=100
    ):
        self.ema_kwargs = {
            'alpha': alpha,
            'bias_correction': bias_correction,
            'truncate': truncate
        }


    def __post_init__(self, related_context, device, logger, model, parent_model):
        super().__post_init__(
            related_context=related_context,
            device=device,
            logger=logger,
            model=model,
            parent_model=parent_model
        )

        # for efficiency reasons, we decorate the model call to compute the one step lookahead
        # logits, so we can have a check what we believed about y for the next step.
        self.parent_model.strategy.get_target_model = override_call_decorator(
            self.parent_model.strategy.get_target_model
        )
        self.parent_model.strategy.get_imputation_augmented_prior = override_call_decorator(
            self.parent_model.strategy.get_imputation_augmented_prior
        )
        if hasattr(self.parent_model.strategy, 'get_error_model'):
            self.parent_model.strategy.get_error_model = override_call_decorator(
                self.parent_model.strategy.get_error_model
            )
        else:
            raise NotImplementedError("Error Model not available")

        self.parent_model.interim_results.update({
            'surprise_logits': [],
            'surprise_x': [],
            'surprises_nll': [],
            'last_step_logits': None,
            'past_x_test': None,

        })

    @staticmethod
    def find_extra_row_index(A, B):
        """
        Find the row not present in B, assuming A has exactly one extra row,
        whose location is unkown.

        :example:
            i=3
            A=torch.randn(10,3)
            B = torch.cat([A.clone()[:i, :], torch.ones(1, 3), A.clone()[i:, :]])
            assert i == find_extra_row_index(B,A)
        :param A:
        :param B:
        :return:
        """
        a = set(tuple(v) for v in A.tolist())
        b = set(tuple(v) for v in B.tolist())
        new_idx = torch.tensor(list(b.difference(a)))

        return torch.where((B == new_idx).all(dim=-1).flatten())[0].item()

    def __call__(self, x_train, x_test, y_train, inc, recompute=False, *args,
                 **kwargs) -> torch.Tensor:
        """
        Here we look at the history of the respective model's predictions
        and find out how each model (target and related) were surprised by the outcome
        """
        interim_results = self.parent_model.interim_results

        target_lookahead = interim_results['last-get_target_model-lookahead']
        imp_aug_lookahead = interim_results['last-get_imputation_augmented_prior-lookahead']
        error_logits = interim_results['last-get_error_model-lookahead']

        # find out where x_train[-1, :, 0] is in x_lookahead

        new_x = self.find_extra_row_index(
            self.parent_model.interim_results['last-x_train'][:,0,:],
            x_train[:,0,:],
        )


        x_lookahead = interim_results['x_lookahead']
        config_idx = (x_train[new_x] == x_lookahead[:, 0, :]).all(dim=-1).flatten()

        try:
            projected_logits, projected_criterion = \
                DistributionConvolver().to(self.device).convolve(
                    A_logits=imp_aug_lookahead[config_idx, : ,:],
                    borders_A=self.model.criterion.borders,
                    B_logits=error_logits[config_idx, : ,:],
                    borders_B=self.parent_model.strategy.err_model.criterion.borders,
                    reverse=False,  # we convolve the error model with the prior
                    target_borders=self.model.criterion.borders,
                    padding=None  # no padding needed here
                )

            # import torch
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            #
            # batch_size = imp_aug_lookahead.shape[1]  # 2 (batch dimension)
            # logit_dim = imp_aug_lookahead.shape[2]  # 1000 (logits dimension)
            #
            # fig, axes = plt.subplots(1, batch_size, figsize=(12, 5), squeeze=False)
            #
            # # Apply softmax function along last dimension (dim=2)
            # imp_aug_soft = torch.softmax(imp_aug_lookahead, dim=2)
            # error_soft = torch.softmax(error_logits, dim=2)
            # proj_soft = torch.softmax(projected_logits, dim=2)
            #
            # err_med = self.parent_model.strategy.err_model.criterion.median(error_logits)
            # aug_med = self.model.criterion.median(imp_aug_lookahead)
            # proj_med =self.model.criterion.median(projected_logits)
            #
            # for b in range(batch_size):
            #     ax = axes[0, b]
            #     # Select the batch distributions (1 x batch x logits)
            #     imp_aug_probs = imp_aug_soft[0, b, :].cpu().numpy()
            #     error_probs = error_soft[0, b, :].cpu().numpy()
            #     proj_probs = proj_soft[0, b, :].cpu().numpy()
            #
            #     # Optional: Use borders as bin edges for histogram
            #     borders = self.model.criterion.borders.cpu().numpy()  # shape: (1001,)
            #
            #     # Plot distribution curves as KDE or histograms
            #     sns.lineplot(x=borders[:-1], y=imp_aug_probs, label='imp_aug', ax=ax, color='blue')
            #     sns.lineplot(x=self.parent_model.strategy.err_model.criterion.borders[:-1],
            #                  y=error_probs, label='error', ax=ax, color='red')
            #     sns.lineplot(x=borders[:-1], y=proj_probs, label='projected', ax=ax, color='green')
            #
            #     ax.set_title(f'Batch {b}')
            #     ax.set_xlabel('Logits bins')
            #     ax.set_ylabel('Probability')
            #     ax.legend()
            #     # ax.set_xlim(0, 1)
            #
            # plt.tight_layout()
            # plt.show()

            last_logits = torch.cat(
                [target_lookahead[config_idx,:,:], projected_logits], dim=1
            ).to(self.device)
            interim_results['surprise_logits'].append(last_logits)
            interim_results['surprise_x'].append(x_train[new_x].cpu().squeeze())

            logits = torch.stack(interim_results['surprise_logits'], dim=0).to(
                self.device).squeeze()

            probs = torch.softmax(logits, dim=-1)

            samples = self.model.criterion.median(logits)

            surprise = torch.stack([
                self.model.criterion(last_logits[:, b, :].squeeze(1), y_train[new_x, :,])
                for b in range(self.num_related + 1)
            ], dim=0).to(self.device).mean(dim=1)

            interim_results['surprises_nll'].append(surprise)

            # FIXME: we can adjust the surprise by how wrong the error model was and by how much
            #  we know better how the error looks like for this point now!
        except Exception as e:
            # this usually is a rare conv error
            log.error(f"Error while computing past surprises: {e}")
            warnings.warn(
                "Error while computing past surprises, returning uniform weights.",
                UserWarning
            )

        surprises = torch.stack(interim_results['surprises_nll'], dim=0).to(
            self.device)

        surprises = ema_conv_causal(surprises, **self.ema_kwargs)

        # we will want to use the history of surprises
        weights = torch.softmax(-surprises[-1], dim=-1)

        self.parent_model.interim_results['last-x_train'] = x_train.cpu()
        return weights

    def plot(self, ax=None, show=True):
        """
        Plot the past surprises.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        past_surprises = torch.stack(self.parent_model.interim_results['surprises_nll'],
                                     dim=0).cpu()
        surprises = ema_conv_causal(past_surprises, **self.ema_kwargs)
        weights = torch.softmax(-surprises, dim=-1).cpu().numpy()
        ax.plot(weights)
        ax.set_title('surprises_nll')
        ax.set_xlabel('Step')
        ax.set_ylabel('weight')

        ax.legend([f'Model {i}' for i in range(self.num_related + 1)])

        if show:
            plt.show()
        return ax


class _CounterfactualPriorWeights(AbstractWeights):
    def __init__(self, counterfactor=None, n_mc=1, counterfit='median', ) -> None:
        self.counterfactor = counterfactor
        self.n_mc = n_mc
        self.counterfit = counterfit

    def __post_init__(self, parent_model, model, criterion, error_model, related_context, logger,
                      device, ):
        super().__post_init__(
            related_context=related_context,
            device=device,
            logger=logger,
            model=model,
            parent_model=parent_model
        )
        self.counterfactor.__post_init__(
            parent_model=self.parent_model,
            model=self.model,
            device=self.device,
            error_model=error_model,
            criterion=criterion,
            logger=self.logger,
            related_context=self.related_context,
        )

    def __call__(self, x_train, y_train, y_error, imputed_y, **kwargs):
        prior_weights, prior_evidence = self.counterfactor(
            x_train=x_train,
            y_error=y_error,
            y_train=y_train,
            related_context=self.related_context,
            imputed_y=imputed_y,
        )
        return prior_weights, prior_evidence


class BMADecayWeights(_CounterfactualPriorWeights):
    """
    Class to compute the BMADecay weights.
    """

    def __init__(self, decay_rate=0.003, *args, **kwargs):
        self.decay_rate = decay_rate
        super().__init__(*args, **kwargs)

    def __post_init__(self, parent_model, model, related_context, error_model, logger, device,
                      **kwargs):
        super().__post_init__(
            parent_model=parent_model,
            model=model,
            related_context=related_context,
            logger=logger,
            device=device,
            criterion=self.model.criterion,
            error_model=error_model,
        )

    @staticmethod
    def constant_exponential(n_target, lambda_=0.001, constant=0):
        effective_n = torch.maximum(torch.tensor(n_target - constant, dtype=torch.float32),
                                    torch.tensor(0.0))
        return 1 - torch.exp(-lambda_ * effective_n)

    def __call__(self, x_train, y_train, *args, **kwargs) -> torch.Tensor:
        """
        Compute the BMADecay weights.
        """
        # Here we assume that the last element in x_train is the current configuration
        # and we compute the BMADecay weights based on the history of predictions
        step = x_train.shape[0]

        prior_weights, prior_evidence = super().__call__(x_train, y_train, *args, **kwargs)

        alpha = self.constant_exponential(
            step,
            lambda_=self.decay_rate,
            constant=self.parent_model.initial_design.size
        )

        weights = torch.cat([torch.tensor([alpha]), (1 - alpha) * prior_weights], dim=0)
        return weights


class BMACVTarget(_CounterfactualPriorWeights):

    def __post_init__(self, parent_model, model, related_context, error_model, logger, device,
                      **kwargs):
        super().__post_init__(
            parent_model=parent_model,
            model=model,
            related_context=related_context,
            logger=logger,
            device=device,
            criterion=self.model.criterion,
            error_model=error_model,
        )

    def __call__(self, x_train, y_train, *args, **kwargs) -> torch.Tensor:
        """
        Compute the BMADecay weights.
        """
        # Here we assume that the last element in x_train is the current configuration
        # and we compute the BMADecay weights based on the history of predictions
        step = x_train.shape[0]

        prior_weights, prior_evidence = super().__call__(x_train, y_train, *args, **kwargs)
        assert x_train.shape[0] >= 10, \
            "BMA with cross-validation requires at least 10 training points."
        target_nll = calc_target_cv_nll(
            x_train.squeeze(1),
            y_train.squeeze(1),
            self.model,
            self.model.criterion,
            splits=5,
            random_state=42
        ).unsqueeze(0).to(self.device)

        weights = torch.softmax(-torch.cat([target_nll, prior_evidence]), dim=-1)

        return weights
