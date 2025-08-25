import torch


class AbstractStrategy:

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

    def __call__(self, x_train, x_test, y_train, inc, *args, **kwargs):
        raise NotImplementedError('This method should be implemented in a subclass.')

    def get_target_model(self, x_test, x_train, y_train):
        return self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )

    def get_prior_model(self, x_test):
        return self.model(
            (
                torch.cat([
                    self.related_context.x,
                    x_test,
                ], dim=0),
                self.related_context.y
            ),
            single_eval_pos=self.related_context.x.shape[0],
        )

    def get_imputation_augmented_prior(self, x_train, x_test, imputed_y):
        return self.model(
            (
                torch.cat([
                    # train
                    self.related_context.x,
                    x_train.repeat(1, self.num_related, 1),

                    # Query
                    x_test.repeat(1, self.num_related, 1),
                ], dim=0),
                torch.cat([self.related_context.y, imputed_y, ], dim=0)
            ),
            single_eval_pos=self.related_context.x.shape[0] + x_train.shape[0],
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(self.num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )
