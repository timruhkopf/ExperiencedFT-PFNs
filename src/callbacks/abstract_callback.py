import abc


class AbstractCallback(abc.ABC):
    __name__ = 'AbstractCallback'
    DEBUG = False
    STOP_AT = -1

    def __init__(self, parent_model, model,  related_context, logger, device, **kwargs):
        self.parent_model = parent_model
        self.logger = logger
        self.device = device
        self.model = model

        self.related_context = related_context
        self.num_related = related_context.x.shape[1]

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.borders_model = self.model.criterion.borders

        if hasattr(self, 'error_model'):
            self.borders_error = self.error_model.criterion.borders

    def step(self, x_train):
        return x_train.shape[0]

    def on_acq_start(self, x_train, y_train, x_test, inc):
        pass

    def on_acq_end_warmstart(self, x_train, y_train, x_test, inc, pi_values):
        pass

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, convolved_logits, convolved_criterion,
            imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        pass

    def on_acq_end_mixture(self, x_train, y_train, x_test, inc, predictions):
        pass
