import torch

from src.model.components.calc_reliability import calc_target_cv_nll
from src.model.weights.abstract_weights import AbstractWeights


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
