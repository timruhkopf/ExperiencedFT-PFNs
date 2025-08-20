import pfns4bo
import torch
import warnings
import logging

from model.components.calc_reliability import calc_target_cv_nll
from model.components.ema_filter import ema_conv_causal
from model.components.error_model_wrapper import WrappedErrorModel
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


class PastSurpriseWeights(AbstractWeights):

    def __init__(
            self,
            err_model='ft-pfn',
            imputation_augmented=False,
            alpha=0.1,
            bias_correction=True,
            truncate=100
    ):
        self.ema_kwargs = {
            'alpha': alpha,
            'bias_correction': bias_correction,
            'truncate': truncate
        }
        self.imputation_augmented = imputation_augmented
        self.err_model = err_model

    def __post_init__(self, related_context, device, logger, model, parent_model):
        super().__post_init__(
            related_context=related_context,
            device=device,
            logger=logger,
            model=model,
            parent_model=parent_model
        )
        self.error_model = WrappedErrorModel(
            target_criterion=self.model.criterion,
            device=self.device,
            error_model=torch.load(pfns4bo.bnn_model, weights_only=False) \
                if self.err_model == 'bnn' else self.model,
        )

        self.parent_model.interim_results['surprise_logits'] = []

    def __call__(self, x_train, x_test, y_train, inc, recompute=False, *args,
                 **kwargs) -> torch.Tensor:
        """
        Here we look at the history of the respective model's predictions
        and find out how each model (target and related) were surprised by the outcome
        """

        # find out which x belongs to the last prediction
        past_x_test = self.parent_model.interim_results['past_x_test']
        config_idx = x_train[-1, :, 0] == past_x_test[:, :, 0]
        config = (x_train[-1, :, 2:] == past_x_test[:, :, 2:]).all(dim=-1)  # ignore fidelity!

        if not config.any():
            log.warning('Detected no matching configuration in past_x_test. ')
            warnings.warn('Detected no matching configuration in past_x_test. ')
            return torch.ones(self.num_related + 1, device=self.device) / (self.num_related + 1)

        if recompute:
            # we need to compute the last prediction

            last_target_prediction = self.model(
                (
                    torch.cat(
                        [x_train[:-1, :, :], past_x_test[config.squeeze(1), :, :]], dim=0
                    ),
                    y_train[:-1, :]
                ),
                single_eval_pos=x_train.shape[0] - 1,
            )
            imputed_y = self.parent_model.interim_results['imputed_y']

            if self.imputation_augmented:
                last_prior_prediction = self.model(
                    (
                        torch.cat([
                            # train
                            self.related_context.x,
                            x_train.repeat(1, self.num_related, 1),

                            # Query
                            past_x_test[config.squeeze(1), :, :].repeat(1, self.num_related, 1),
                        ], dim=0),
                        torch.cat([self.related_context.y, imputed_y, ], dim=0)
                    ),
                    single_eval_pos=self.related_context.x.shape[0] + x_train.shape[0],

                )
            else:
                last_prior_prediction = self.model(
                    (torch.cat(
                        [self.related_context.x,
                         past_x_test[config.squeeze(1), :, :].repeat(1, self.num_related, 1)],
                        dim=0),
                     self.related_context.y),
                    single_eval_pos=self.related_context.x.shape[0],
                )

            if self.err_model == 'bnn':
                x = x_train[:, :, 1:]
                x_t = past_x_test[config.squeeze(1), :, 1:]
            else:
                x = x_train
                x_t = past_x_test[config.squeeze(1), :, :]

            y_error = y_train.repeat(1, self.num_related) - imputed_y
            projected_logits, error_logits, convolved_criterion = (
                self.error_model.convolve_probs_with_error(
                    logits=last_prior_prediction,
                    logits_borders=self.model.criterion.borders,
                    x_train=x.repeat(1, self.num_related, 1),
                    x_test=x_t.repeat(1, self.num_related, 1),
                    y_error=y_error
                ))

            projected_logits = project_probs_to_new_grid(
                projected_logits, convolved_criterion.borders, self.model.criterion.borders,
                return_logits=True
            )

            last_prediction = torch.cat(
                [last_target_prediction, projected_logits], dim=1
            )

        else:
            last_step_predictions = self.parent_model.interim_results['last_step_predictions']
            last_prediction = last_step_predictions[config.flatten(), :, :]

        # for plotting purposes
        self.parent_model.interim_results['surprise_logits'].append(last_prediction)
        self.parent_model.interim_results['error_logits'] = error_logits

        surprise = torch.stack([
            self.model.criterion(last_prediction[:, b, :].squeeze(1), y_train[-1, :,])
            for b in range(self.num_related + 1)
        ], dim=0).to(self.device).mean(dim=1)

        if 'past_surprises' not in self.parent_model.interim_results:
            self.parent_model.interim_results['past_surprises'] = []

        self.parent_model.interim_results['past_surprises'].append(surprise)

        # FIXME: we can adjust the surprise by how wrong the error model was and by how much
        #  we know better how the error looks like for this point now!
        surprises = torch.stack(self.parent_model.interim_results['past_surprises'], dim=0).to(
            self.device)

        surprises = ema_conv_causal(surprises, **self.ema_kwargs)

        # we will want to use the history of surprises
        weights = torch.softmax(-surprises[-1], dim=-1)
        return weights

    def plot(self, ax=None, show=True):
        """
        Plot the past surprises.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        past_surprises = torch.stack(self.parent_model.interim_results['past_surprises'],
                                     dim=0).cpu()
        surprises = ema_conv_causal(past_surprises, **self.ema_kwargs)
        weights = torch.softmax(-surprises, dim=-1).cpu().numpy()
        ax.plot(weights)
        ax.set_title('Past Surprises')
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
