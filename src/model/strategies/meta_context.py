import torch

from model.components.imputor import PriorImputer
from model.strategies.abstract_strategy import AbstractStrategy


class PFNV2(AbstractStrategy):

    def __init__(
            self, model, criterion, logger,
            related_task_data,
            min_context_size,
            imputation_mode='median',
            incumbent_calculation='imputation-only',
            callbacks=(),
    ):
        self.model = model
        self.criterion = criterion
        self.logger = logger
        self.related_task_data = related_task_data
        self.min_context_size = min_context_size

        self.related_task_data = related_task_data

        self.min_context_size = min_context_size
        self.incumbent_calculation = incumbent_calculation
        self.imputer = PriorImputer(
            model, criterion, imputation_mode
        )

        self.callbacks = callbacks

        self.initialized = False  # initializing the related_context during first call to meet
        # flipping needs
        self.num_related = None
        self.related_context = None

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        pass


    def __call__(
            self,
            imputed_y,
            related_context,
            x_train,
            x_test,
            y_train,
            inc
    ):
        step = x_train.shape[0]

        imputed_y_test = self.imputer(
            x_train=related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=related_context.y)

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
            single_eval_pos=related_context.x.shape[0],
            # src_key_padding_mask=padding_mask
        )

        # TODO find a weighing!
        return target_logits.mean(dim=1)


class PPFNV3(AbstractStrategy):

    def __init__(self, model, criterion, logger,
                 related_task_data, min_context_size, imputation_mode='median',
                 incumbent_calculation='imputation-only'
                 ):
        super().__init__(model, criterion, logger,
                         related_task_data, min_context_size, imputation_mode,
                         incumbent_calculation)
        self.num_related = len(related_task_data)

    def mixture_strategy(
            self,
            imputed_y,
            related_context,
            x_train,
            x_test,
            y_train,
            inc
    ):
        step = x_train.shape[0]

        imputed_y_test = self.imputer(
            x_train=related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=related_context.y)

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')

        # here all the priors are directly concatenated to the HP dimensions!
        features = torch.concat([x_train.unsqueeze(1), imputed_y.unsqueeze(1)], dim=-1)
        features_test = torch.concat([x_test.unsqueeze(1), imputed_y_test.unsqueeze(1)], dim=-1)
        target_logits = self.model(
            (
                torch.cat([features, features_test]),
                y_train
            ),
            single_eval_pos=related_context.x.shape[0],
            # src_key_padding_mask=padding_mask
        )

        return target_logits.mean(dim=1)


