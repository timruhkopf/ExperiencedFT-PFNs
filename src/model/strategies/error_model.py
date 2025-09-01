from functools import partial

import pfns4bo
import torch

from model.probability_conv.convolver import DistributionConvolver
from model.strategies.abstract_strategy import AbstractStrategy


class ErrorModelStrategies(AbstractStrategy):
    __name__ = "ErrorModelStrategies"

    def __post_init__(self, criterion, **kwargs):
        super().__post_init__(**kwargs)

        self.err_model = torch.load(pfns4bo.bnn_model, weights_only=False)
        self.err_model.to(self.device)

    def get_error_model(self, x_train, y_error, x_test, padding=None):
        """
        Get the error model predictions for the given training and test data.
        Args:
            x_train: training data
            y_error: error values for training data
            x_test: test data
            padding: optional padding mask
        Returns:
            error_probs_kernel, kernel_grid, error_logits
        """
        msg = "Error model expects self.num_related  tasks"
        assert x_train.shape[1] == self.num_related, msg
        assert x_test.shape[1] == self.num_related, msg
        assert y_error.shape[1] == self.num_related, msg

        return self.err_model(
            (
                torch.cat([x_train[:,:, 1:], x_test[:,:, 1:]], dim=0),
                y_error
            ),
            single_eval_pos=x_train.shape[0],
        )

    def __call__(self, x_train, x_test, y_train, inc):

        step = x_train.shape[0]
        imputed_y = self.parent_model.interim_results['imputed_y']

        # TODO the following two forwards can be batched together with
        #  appropriate padding masks. this will save wallclock time
        # (Collect target task logits) --------------------------------
        # CAREFUL: here we also do the query forward for the evidence

        # Create a partial that fixes all arguments except x_test
        target_model = partial(
            self.get_target_model,
            x_train=x_train, y_train=y_train
        )
        target_logits = target_model(x_test=x_test)

        # (Collect prior logits) -------------------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        imputation_augmented_prior = partial(
            self.get_imputation_augmented_prior,
            x_train=x_train, imputed_y=imputed_y
        )
        imputation_augmented_prior_logits = imputation_augmented_prior(x_test=x_test)


        # (Project prior logits into target task) --------------------------
        # Here we take the predicted prior logits of the x_test and need to adjust them
        # according to the error logits. -- which tell us how to shift the distribution
        # (median) and given the shift, how to adjust the probability mass.
        y_error = y_train.repeat(1, self.num_related) - imputed_y

        error_model = partial(
            self.get_error_model,
            x_train=x_train.repeat(1, self.num_related, 1), y_error=y_error,
        )
        error_logits = error_model(x_test=x_test.repeat(1, self.num_related, 1)) # unnormalized


        projected_logits, projected_criterion = \
            DistributionConvolver().to(self.device).convolve(
            A_logits=imputation_augmented_prior_logits,
            borders_A=self.model.criterion.borders,
            B_logits=error_logits,
            borders_B=self.err_model.criterion.borders,
            reverse=False,  # we convolve the error model with the prior
            target_borders=self.model.criterion.borders,
            padding=None  # no padding needed here
        )

        self.parent_model.interim_results.update({
            'target_model': target_model,
            'imputation_augmented_prior': imputation_augmented_prior,
            'raw_error_model': error_model,
            'raw_error_criterion': self.err_model.criterion,
            'prior_model': self.get_prior_model,
            'y_error': y_error,
            'imputed_y': imputed_y,
        })  # for plotting purposes

        predictions = torch.cat(
            [target_logits, projected_logits], dim=1
        ).to(self.device)
        self.parent_model.interim_results.update({
            'last_step_logits': predictions,
            'past_x_test': x_test,
        })

        weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=False)
        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item()
                for i, w in enumerate(weights)}},
        )

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return (predictions * weights.unsqueeze(1)).mean(dim=1)

