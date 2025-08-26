import torch

from src.model.weights.abstract_weights import AbstractWeights


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
