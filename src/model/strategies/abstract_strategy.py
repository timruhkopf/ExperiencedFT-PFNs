import torch


class AbstractStrategy:

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        self.parent_model = parent_model
        self.model = model
        self.model.to(device)
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.callbacks = callbacks
        self.logger = logger
        self.device = device
        self.kwargs = kwargs

    def __call__(self, x_train, x_test, y_train, inc, *args, **kwargs):
        raise NotImplementedError('This method should be implemented in a subclass.')

    def get_target_model(self, x_test, x_train, y_train):
        """

        :param x_test:
        :param x_train:
        :param y_train:
        :return:
        """
        return self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )

    def get_prior_model(self, x_test):
        """
        Evaluate the logits for test points under the prior unconditional on currently
        evaluated data.
        :param x_test: T', 1, D
        :return: logits T', B, D
        """
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
        """
        We take the prior (related task data x,y ) and then collect for the target data (x_train)
        imputations y and augment the training set by them as if they were evaluated;
        This helps keep the prior up-to-date and relevant wrt. knowing about where we already
        have evaluated.
        :param x_train: T, 1, D
        :param x_test: T', 1, D
        :param imputed_y: T, 1
        :return: logits T', B, D; i.e. the logits for x_test based on each of the B priors
        """
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
