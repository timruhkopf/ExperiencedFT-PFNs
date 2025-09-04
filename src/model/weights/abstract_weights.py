import torch


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

