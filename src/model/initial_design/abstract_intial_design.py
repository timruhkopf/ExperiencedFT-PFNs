
class AbstractInitialDesign:
    def __init__(self, size,):
        self.size = size

        self.model= None
        self.related_context = None
        self.num_related = None

    def __post_init__(self, parent_model, model, related_context, logger, device, callbacks=[]):
        self.parent_model = parent_model
        self.model = model
        self.related_context = related_context
        self.num_related = related_context.x.shape[1] if related_context is not None else None
        self.logger = logger
        self.device = device
        self.callbacks = callbacks

    def __call__(self, x_train, x_test, y_train, inc):
        """
        This method should be implemented by subclasses to provide the initial design
        for the optimization process.

        :param step: The current step in the optimization process.
        :return: A tensor containing the initial design points.
        """
        raise NotImplementedError("Subclasses must implement this method.")
