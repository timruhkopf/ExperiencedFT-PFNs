

class AbstractStrategy:
    def __init__(self):
        pass

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        self.parent_model = parent_model
        self.model = model
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.callbacks = callbacks
        self.logger = logger
        self.device = device
        self.kwargs = kwargs

    def __call__(self, x_train, x_test, y_train, inc):
        raise NotImplementedError('This method should be implemented in a subclass.')
