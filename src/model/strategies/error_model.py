from functools import partial

import pfns4bo
import torch

from model.components.calc_reliability import calc_target_cv_nll
from model.components.counterfact import Counterfactor
from model.components.error_model_wrapper import WrappedErrorModel
from model.probability_conv.convolver import DistributionConvolver
from model.probability_conv.map_binnings import project_probs_to_new_grid
from model.strategies.abstract_strategy import AbstractStrategy


debug = False

class ErrorModelStrategies(AbstractStrategy):

    def __init__(self, model_avg, counterfactor):
        self.model_avg = model_avg
        self.counterfactor = counterfactor  # type: Counterfactor
        self.__name__ = f"ErrorModelStrategies-{model_avg}"

    def __post_init__(self, criterion, **kwargs):
        super().__post_init__(**kwargs)

        self.err_model = error_model=torch.load(pfns4bo.bnn_model, weights_only=False)
        self.error_model = WrappedErrorModel(
            target_criterion=criterion,
            device=self.device, error_model=self.err_model
        )

        self.counterfactor.__post_init__(
            parent_model=self.parent_model,
            model=self.model,
            device=self.device,
            error_model=self.error_model,
            criterion=criterion,
            logger=self.logger,
            related_context=self.related_context,
        )

        self.parent_model.interim_results.update(dict(past_surprises=[]))

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

        if self.model_avg == 'ppd_mixture_eqw':
            query_size = x_test.shape[0]
            predictions = torch.concat([
                target_logits[:query_size],
                imputation_augmented_prior_logits[:query_size]
            ], dim=1)

            weights = torch.ones(self.num_related) / self.num_related
            for callback in self.callbacks:
                callback.on_final_weights(predictions, weights)

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

        for callback in self.callbacks:
            callback.on_trained_ppds(
                target_logits, imputation_augmented_prior_logits, error_logits, projected_logits,
                projected_criterion,
                imputed_y, y_error,
                x_train, y_train, x_test, inc
            )

        projected_logits = project_probs_to_new_grid(
            projected_logits,
            projected_criterion.borders,
            self.model.criterion.borders,
            return_logits=True
        )

        predictions = torch.cat(
            [target_logits, projected_logits], dim=1
        ).to(self.device)

        self.parent_model.interim_results.update({
            'last_step_predictions': predictions,
            'past_x_test': x_test,
        })



        if self.model_avg in ['prior-mixture', 'bma-decay', 'bma-cv-target']:
            prior_weights, prior_evidence = self.counterfactor(
                x_train=x_train,
                y_error=y_error,
                y_train=y_train,
                related_context=self.related_context,
                imputed_y=imputed_y,
            )

        # WEIGHTS --------------------------------------------------------------
        if self.model_avg == 'project_eqw':  # equally weighted average
            # here we simply average the predictions over the projected prior with convolved
            # related tasks
            weights = torch.ones(self.num_related + 1) / (self.num_related + 1)

        if self.model_avg == 'prior-mixture':
            # here we simply average the predictions over the related tasks
            weights = torch.ones(self.num_related +1)
            weights[0] = 0
            weights[1:] = prior_evidence

        if self.model_avg == 'bma-decay':
            # In this case we do
            # \alpha(step) p(y|M_{\tau^*}) + (1-\alpha(step)) p_BMA(y|{M_\tau}_{\tau
            # \neq \tau^*})
            # where \tau^* is the target task and \alpha(step) is an exponential decay function
            # that decays with the number of steps.

            alpha = constant_exponential(
                step,
                lambda_=self.kwargs.get('decay_rate', 0.003),
                constant=self.parent_model.initial_design.size
            )

            # if debug:
            #     import numpy as np
            #     min_context_size = 10  # constant parameter
            #     lambda_ = 0.003  # decay rate
            #     steps = np.arange(0, 1001)  # 0-1000 inclusive
            #     alpha = constant_exponential(steps, lambda_, min_context_size)
            #     plt.figure(figsize=(10, 6))
            #     plt.plot(steps, alpha)
            #     plt.title('Alpha decay over steps')
            #     plt.xlabel('Step')
            #     plt.ylabel('Alpha')
            #     plt.grid(True)
            #     plt.show()

            weights = torch.cat([torch.tensor([alpha]), (1 - alpha) * prior_weights], dim=0)


        if self.model_avg == 'bma-cv-target':
            assert x_train.shape[0] >= 10, \
                "BMA with cross-validation requires at least 10 training points."
            target_nll = calc_target_cv_nll(
                x_train.squeeze(1),
                y_train.squeeze(1),
                self.model,
                self.model.criterion,
                splits=5,
                random_state=42
            ).unsqueeze(0).to(self.device)

            weights = torch.softmax(-torch.cat([target_nll, prior_evidence]), dim=-1)


        if self.model_avg == 'bma':
            raise NotImplementedError(
                "BMA model averaging is not implemented yet, "
                "we didn't find a way to calculate the evidence p(H | M_{\\tau^*})."
            )

        if self.model_avg == 'past_surprise':
            # Here we look at the history of the respective model's predictions
            # and find out how each model (target and related) were surprised by the outcome
            config_idx = x_train[-1, :, 0] == self.parent_model.interim_results['past_x_test'][:, :, 0]
            config = (x_train[-1, :, 2:] == self.parent_model.interim_results['past_x_test'][:,
            :, 2:]).all(dim=-1)  # ignore fidelity!
            last_prediction = self.parent_model.interim_results['last_step_predictions'][
                config_idx.flatten() & config.flatten(), :, :]

            surprise = torch.stack([
                self.model.criterion(last_prediction[:, b, :].squeeze(1), y_train[-1, :,])
                for b in range(self.num_related + 1)
            ], dim=0).to(self.device)

            self.parent_model.interim_results['past_surprises'].append(surprise)

            # FIXME: we can adjust the surprise by how wrong the error model was and by how much
            #  we know better how the error looks like for this point now!
            surprise = torch.stack(self.parent_model.interim_results['past_surprises'], dim=0).mean(
                dim=0)

            # we will want to use the history of surprises
            weights = torch.softmax(-surprise.flatten(), dim=-1)

        if self.model_avg == 'past_suprise_updated_error':
            # we take the best possible prediction, by retrospectively updating the predictions
            # this will improve the prior's projections and we will get a better sense
            # for whether the prior was surprised by the outcome - this will implicitly grant
            # access to the future and in turn will adversely bias
            # against the target task predictions, because it won't be updated
            pass

        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item()
                for i, w in enumerate(weights)}},
        )

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return (predictions * weights.reshape(1, self.num_related +1 , 1)).sum(dim=1)



def constant_exponential(n_target, lambda_=0.001, constant=0):
    effective_n = torch.maximum(torch.tensor(n_target - constant, dtype=torch.float32),
                                torch.tensor(0.0))
    return 1 - torch.exp(-lambda_ * effective_n)
