from functools import partial

import torch

from model.probability_conv.convolver import DistributionConvolver
from model.strategies.abstract_strategy import AbstractStrategy


class SplitMetaContextStrategy(AbstractStrategy):
    __name__ = "SplitMetaContextStrategy"

    def __init__(self, imputer, model_avg='mean'):
        self.imputer = imputer
        self.model_avg = model_avg

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        super().__post_init__(
            parent_model=parent_model,
            model=model,
            related_context=related_context,
            callbacks=callbacks,
            logger=logger,
            device=device,
            **kwargs
        )

        self.imputer.__post_init__(
            parent_model=self,
            model=self.model,
            criterion=self.model.criterion,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device
        )

    def get_prior_augmented_target_model(self, x_train, x_test, y_train):
        # how does x_test look under the related tasks?
        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y)

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')

        imputed_y = self.parent_model.interim_results['imputed_y']

        # Here we concatenate one imputed prior value to the HP dimension, but repeat it for all related tasks.
        features = torch.concat(
            [x_train.repeat(1, self.num_related, 1), imputed_y.unsqueeze(-1)], dim=-1
        )
        features_test = torch.concat(
            [x_test.repeat(1, self.num_related, 1), imputed_y_test.unsqueeze(-1)], dim=-1
        )
        return self.model(
            (
                torch.cat([features, features_test]),
                y_train.repeat(1, self.num_related)
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )

    def __call__(self, x_train, x_test, y_train, inc):
        """

        This strategy uses imputed y values under the prior and concatenates them to the
        x_train and x_test, but it only ever uses a single related task at a time.
        We abuse the batch dim to do it concurrently for all related tasks.

        """

        step = x_train.shape[0]

        target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
        target_logits = target_model(x_test=x_test)

        prior_augmented_target_model = partial(
            self.get_prior_augmented_target_model,
            x_train=x_train, y_train=y_train
        )
        prior_logits = prior_augmented_target_model(x_test=x_test)

        predictions = torch.cat((target_logits, prior_logits), dim=1)

        self.parent_model.interim_results.update({
            'prior_augmented_target_model': prior_augmented_target_model,
            'target_model': target_model,
            'past_x_test': x_test,
            'logits': predictions,
            # 'error_logits': target_logits,  # for plotting purposes
        })

        # TODO find a weighing!
        weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)

        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item()
                for i, w in enumerate(weights)}},
        )

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return (predictions * weights.unsqueeze(1)).sum(dim=1)


class JointMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointMetaContextStrategy"

    def __call__(self, x_train, x_test, y_train, inc):
        """
        Main idea behind this method is that we can use all of the related task y values (imputed
        for the x_test values of course) and concatenate them to the hp dimension.
        ideally, the model will learn to recognize whether the prior y dims are relevant.
        They are basically a probing feature that may or may not be relevant for the target task.


        # FIXME: DISADVANTAGE: WE CAN ONLY USE 10 DIMENSIONS IN TOTAL FOR THE INPUT FEATURES.
        # GIVEN THAT THE HP ALSO OCCUPY THIS SPACE, WE CAN ONLY USE 10 - HP_DIMENSIONS
        # FOR THE RELATED TASKS.

        :param x_train:
        :param x_test:
        :param y_train:
        :param inc:
        :return:
        """
        step, _, hp = x_train.shape

        # sample without the available priors based on their past surprise:

        avail_space = 10 - hp
        assert avail_space > 0, \
            (f'PFN input restriction violated: n_hp:{x_train.shape[-1]}. There are'
             '{self.num_related} related tasks, but they don\'t fit into the hp dim')
        if avail_space < self.num_related:
            weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)
            prior_weights = weights[1:]

            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )

            # now sample:
            prior_idx = torch.multinomial(
                prior_weights, num_samples=min(avail_space, self.num_related),
                replacement=False
            )
        else:
            prior_idx = torch.arange(self.num_related, device=self.device)

        imputed_y = self.parent_model.interim_results['imputed_y']

        imputed_y_test = self.imputer(
            x_train=self.related_context.x[:, prior_idx, :],
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y[:, prior_idx])

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

        self.parent_model.interim_results['logits'] = target_logits  # for plotting purposes

        # TODO past surprise weighing by prior?

        for callback in self.callbacks:
            callback.on_final_weights(target_logits, weights)
        return target_logits


class JointNoHPMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointNoHPMetaContextStrategy"

    def __init__(self, imputer, prior_only=False, **kwargs):
        super().__init__(imputer=imputer, **kwargs)
        self.prior_only = prior_only


    def __call__(self, x_train, x_test, y_train, inc):
        """
        Here, we only will use the imputed prior y values as input to the model.
        We can augment the model prediction with that of a meta-unaware target model,
        solely based on the hp values.
        """

        step, _, hp = x_train.shape

        # sample without the available priors based on their past surprise:

        avail_space = 10
        assert avail_space > 0, \
            (f'PFN input restriction violated: n_hp:{x_train.shape[-1]}. There are'
             '{self.num_related} related tasks, but they don\'t fit into the hp dim')
        if avail_space < self.num_related:
            weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)
            prior_weights = weights[1:]

            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )

            # now sample:
            # TODO instead of subsampling just batch the model calls over different contexts--
            # and then measure their respective surprises
            prior_idx = torch.multinomial(
                prior_weights, num_samples=min(avail_space, self.num_related),
                replacement=False
            )
        else:
            prior_idx = torch.aragne(self.num_related, device=self.device)
            weights = torch.ones(self.num_related + 1, device=self.device) / (self.num_related + 1)

        # Collect the y (train/test) values under the priors as "landmarking" features:
        imputed_y = self.parent_model.interim_results['imputed_y']
        imputed_y = imputed_y[:, prior_idx]
        imputed_y = imputed_y.reshape(imputed_y.shape[0], 1 - 1)

        imputed_y_test = self.imputer(
            x_train=self.related_context.x[:, prior_idx, :],
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y[:, prior_idx])
        imputed_y_test = imputed_y_test[:, prior_idx]
        imputed_y_test = imputed_y_test.reshape(imputed_y_test.shape[0], 1, -1)

        imputed_y = torch.cat([x_train[..., :2], imputed_y], dim=-1)  # prepend the id and fidelity dim
        imputed_y_test = torch.cat([x_test[..., :2], imputed_y_test], dim=-1)

        # fit the model for the current data only as a function of the imputed prior values:
        prior_logits = self.model(
            (
                torch.cat([imputed_y, imputed_y_test]),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )
        predictions = prior_logits

        if not self.prior_only:
            # collect the meta-unaware target model as well:
            target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
            target_logits = target_model(x_test=x_test)

            # finally convolve the two predictions:
            convolved_logits = DistributionConvolver().to(self.device).convolve(
                A_logits=target_logits,
                borders_A=self.model.criterion.borders,
                B_logits=prior_logits,
                borders_B=self.model.criterion.borders,
                reverse=False,  # we convolve the error model with the prior
                target_borders=self.model.criterion.borders,
                padding=None  # no padding needed here
            )
            predictions = convolved_logits

        self.parent_model.interim_results['logits'] = predictions  # for plotting purposes

        # TODO past surprise weighing by prior?

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return predictions


class JointBatchedMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointBatchedMetaContextStrategy"

    def __init__(self, imputer, D=10, convolve=False,  **kwargs):
        """
        D: max tasks per inner batch dimension (fixed at <= 10)
        """
        super().__init__(imputer=imputer, **kwargs)
        self.convolve = convolve

        assert D <= 10, "D must be <= 10 due to PFN input restrictions"
        self.D = D

    def get_prior_augmented_target_model(self, x_train, x_test, y_train):

        num_rel = self.num_related
        # choose B and D such that B*D >= num_rel and D <= self.D
        D = min(self.D, num_rel)
        B = (num_rel + D - 1) // D  # ceil division
        pad = B * D - num_rel


        if pad > 0:
            raise NotImplementedError()

            excess_imputed_y = imputed_y[:, B*D: , :].view(imputed_y.shape[0], -1)
            # todo a separate fwd with different dimensionality on the excess tasks

        # get imputed prior values and pad if needed
        imputed_y = self.parent_model.interim_results['imputed_y']  # (T, num_rel)
        imputed_y = imputed_y[:,:B*D].view(imputed_y.shape[0], B, D)

        # imputed test ys
        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, num_rel, 1),
            y_train=self.related_context.y
        )  # (T, num_rel)

        imputed_y_test = imputed_y_test[:, :B * D].view(imputed_y_test.shape[0], B, D)

        # prepend the id and fidelity dim of x_train and x_test respectively:
        feature_train = torch.cat([x_train.repeat(1, B, 1), imputed_y], dim=-1)  # (T, B,
        # hp+1)
        feature_test = torch.cat([x_test.repeat(1, B, 1), imputed_y_test],
                                   dim=-1)  # (testT, B, hp+1)

        # now model input is batched across (B,D)
        prior_logits = self.model(
            (
                torch.cat([feature_train, feature_test]),  # (T+testT, B, D)
                y_train.repeat(1, B)  # (T, B)
            ),
            single_eval_pos=y_train.shape[0],
        )

        return prior_logits

    def __call__(self, x_train, x_test, y_train, inc):
        step, _, hp = x_train.shape

        prior_augmented_target_model = partial(
            self.get_prior_augmented_target_model,
            x_train=x_train, y_train=y_train
        )
        prior_logits = prior_augmented_target_model(x_test=x_test)

        target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
        target_logits = target_model(x_test=x_test)

        predictions = torch.cat((target_logits, prior_logits), dim=1)

        # compute weights ONCE
        weights = self.parent_model.weights(
            x_train, x_test, y_train, inc, recompute=True
        )

        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item() for i, w in enumerate(weights)}}
        )


        self.parent_model.interim_results.update({
            'prior_augmented_target_model': prior_augmented_target_model,
            'target_model': target_model,
            'past_x_test': x_test,
            'logits': predictions,
        })

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)


        if self.convolve:
            prior_weights = (weights[1:] / weights[1:].sum()) # normalize to sum to 1
            # finally convolve the two predictions:
            convolved_logits, convolved_criterion = DistributionConvolver().to(
                self.device).convolve(
                A_logits=target_logits,
                borders_A=self.model.criterion.borders,
                B_logits=(prior_logits * prior_weights.unsqueeze(1)).sum(dim=1, keepdim=True ),
                borders_B=self.model.criterion.borders,
                reverse=False,  # we convolve the error model with the prior
                target_borders=self.model.criterion.borders,
                padding=None  # no padding needed here
            )

            return convolved_logits

        else:
            return (predictions * weights.unsqueeze(1)).sum(dim=1)

        # self.parent_model.interim_results['logits'] = predictions
