import torch

from model.strategies.abstract_strategy import AbstractStrategy


class MetaContextStrategy(AbstractStrategy):
    __name__ = "MetaContextStrategy"

    def __init__(self, imputer):
        self.imputer = imputer


    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        self.parent_model = parent_model
        self.model = model
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.logger = logger
        self.device = device

        self.imputer.__post_init__(
            parent_model=self,
            model=self.model,
            criterion=self.model.criterion,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device
        )
    def __call__(self, x_train, x_test, y_train, inc):
        """
        Main idea behind this method is that we can use all of the related task y values (imputed
        for the x_test values of course) and concatenate them to the hp dimension.
        ideally, the model will learn to recognize whether the prior y dims are relevant.
        They are basically a probing feature that may or may not be relevant for the target task.

        # FIXME: DISADVANTAGE: WE CAN ONLY USE 10 DIMENSIONS IN TOTAL FOR THE INPUT FEATURES.
        # GIVEN THAT THE HP ALSO OCCUPY THIS SPACE, WE CAN ONLY USE 10 - HP_DIMENSIONS
        # FOR THE RELATED TASKS.

        """
        imputed_y = self.parent_model.interim_results['imputed_y']
        step = x_train.shape[0]

        # how does x_test look under the related tasks?
        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y)

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')


        # Here we concatenate one imputed prior value to the HP dimension, but repeat it for all related tasks.
        features = torch.concat(
            [x_train.repeat(1, self.num_related, 1), imputed_y.unsqueeze(-1)], dim=-1
        )
        features_test = torch.concat(
            [x_test.repeat(1, self.num_related, 1), imputed_y_test.unsqueeze(-1)], dim=-1
        )
        target_logits = self.model(
            (
                torch.cat([features, features_test]),
                y_train.repeat(1, self.num_related)
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )

        # TODO find a weighing!
        return target_logits.mean(dim=1)


class SplitMetaContextStrategy(MetaContextStrategy):

    def __call__(self, x_train, x_test, y_train, inc):
        """
        This strategy also uses inputed y values under the prior and concatentates them to the
        x_train and x_test, but it only ever uses a single related task at a time.
        We abuse the batch dim to do it concurrently for all related tasks.

        :param x_train:
        :param x_test:
        :param y_train:
        :param inc:
        :return:
        """
        step = x_train.shape[0]
        imputed_y = self.parent_model.interim_results['imputed_y']

        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y)

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')

        # here all the priors are directly concatenated to the HP dimensions!
        features = torch.concat([x_train, imputed_y.unsqueeze(1)], dim=-1)
        features_test = torch.concat([x_test, imputed_y_test.unsqueeze(1)], dim=-1)
        target_logits = self.model(
            (
                torch.cat([features, features_test]),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )

        # TODO past surprise weighing by prior?
        return target_logits.mean(dim=1)
