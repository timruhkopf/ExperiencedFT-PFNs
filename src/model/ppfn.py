import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluation.callbacks import CallbackErrorModelRMSE
from ifbo import BarDistribution
from ifbo.transformer import TransformerModel
from model.calc_reliability import calc_target_cv_nll
from model.counterfact import Counterfactor
from model.error_model_wrapper import WrappedErrorModel
from model.imputor import Imputer

from src.model.abstractmodel import AbstractModel
from src.utils.dotdict import DotDict
from utils.filelogger import BufferedFileLogger
import pfns4bo

log = logging.getLogger(__name__)

# fixme: move all the plots and logging metrics into an (optional) callback

debug = False


class PPFN(AbstractModel):
    __name__ = "pPFN"

    def __init__(self, model, criterion: BarDistribution, logger,
                 related_task_data, min_context_size, imputation_mode='median',
                 incumbent_calculation='imputation-only', flippable_related=False,
                 model_avg='bma',
                 device=None, verbose=True,
                 callbacks=[
                     CallbackErrorModelRMSE,
                     # CallbackAnytime
                     # CallbackImputationDiff
                     # Callback1DProjection,
                 ],
                 **kwargs):

        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model
        self.model.eval()

        # $HOME/anaconda3/envs/ft-pfn-experimental/lib/python3.10/site-packages/pfns4bo/final_models
        # /hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt.gz
        self.error_model = WrappedErrorModel(
            target_criterion=self.model.criterion,
            device=device, error_model=torch.load(pfns4bo.bnn_model, weights_only=False)
        )

        self.criterion = criterion if criterion is not None else self.model.criterion
        self.updated_prior_validation = Counterfactor(
            model=self.model,
            device=device,
            error_model=self.error_model,
            criterion=self.criterion,
            logger=logger,
            counterfit=kwargs.get('counterfit', 'median'),
            num_related=related_task_data.x.shape[1],
            n_mc=kwargs.get('n_mc', 10)
            # number of Monte Carlo samples if mc counterfitting is used
        )
        self.imputer = Imputer(
            model=self.model,
            criterion=self.model.criterion,
            imputation_mode=imputation_mode
        )

        self.logger = logger
        self.device = device

        self.related_task_data = related_task_data

        self.min_context_size = min_context_size
        self.incumbent_calculation = incumbent_calculation
        self.flippable_related = flippable_related

        self.model_avg = model_avg

        self.verbose = verbose
        self.kwargs = kwargs

        self.callbacks = callbacks

        log.info(f'Instantiated {self.__name__}')

        self.initialized = False  # initialzing the related_context during first call to meet
        # flipping needs
        self.num_related = None
        self.related_context = None

        self.past_surprises = []

    def _initialize(self, related_context_data, minimize):
        related_context_x = related_context_data.x
        related_context_y = related_context_data.y
        padding_mask = related_context_data.padding_mask

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)

        related_context_x = related_context_x.to(self.device)
        related_context_y = related_context_y.to(self.device)

        self.related_context = DotDict({
            'x': related_context_x,
            'y': related_context_y,
            'padding_mask': padding_mask
        })

        self.num_related = self.related_context.x.shape[1]
        cbs = []
        for callback in self.callbacks:
            cbs.append(
                callback(
                    self.model,
                    self.error_model,
                    self.related_task_data,
                    self.logger,
                    self.device,
                    **self.kwargs
                )
            )
        self.callbacks = cbs

    def _preprocess(self, x_train, y_train, x_test, inc, minimize=True):

        if not self.initialized:
            self._initialize(self.related_task_data, minimize)
            self.initialized = True

        y_train = y_train.to(self.device)
        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        inc = inc.to(self.device)

        y_train = y_train.unsqueeze(1)
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

        if torch.any(x_test > 999.):
            log.warning(f"Query points x_test contain values > 999: {x_test[x_test > 999.]}")
            x_test = torch.clamp(x_test, max=999.)
        if torch.any(x_train > 999.):
            log.warning(f"Training points x_train contain values > 999: {x_train[x_train > 999.]}")
            x_train = torch.clamp(x_train, max=999.)

            # (Adjust searchspaces) ------------------------------------------------
        # in case the search spaces are supersets of each other, we need to augment the
        # x_train data to match the related context (we need to drop the dim later for the
        # acquisition function to not notice)
        if x_train.shape[-1] < self.related_context.x.shape[-1]:
            diff = self.related_context.x.shape[-1] - x_train.shape[-1]
            placeholder = self.related_context.x[:, :, -diff:].mean(dim=1).mean(dim=0)
            x_train = torch.cat([
                x_train,
                placeholder.repeat(x_train.shape[0], 1).unsqueeze(1)
            ], dim=-1).to(self.device)

            x_test = torch.cat([
                x_test,
                placeholder.repeat(x_test.shape[0], 1).unsqueeze(1)
            ], dim=-1).to(self.device)

        return x_train, y_train, x_test, inc

    @torch.no_grad()
    def get_ei(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        step = x_train.shape[0]
        x_train, y_train, x_test, inc = \
            self._preprocess(x_train, y_train, x_test, inc, minimize=minimize)
        # (Impute related tasks) -----------------------------------------------

        imputed_y = self.imputer(
            x_train=self.related_context.x,
            x_test=x_train.repeat(1, self.num_related, 1),
            y_train=self.related_context.y
        )

        if step < self.min_context_size:
            # FIXME: EI values!
            return self.warmstart(
                imputed_y,
                self.related_context,
                x_train,
                x_test,
                y_train,
                inc
            )
        else:
            bma_predictions = self.mixture_strategy(
                imputed_y,
                self.related_context,
                x_train,
                x_test,
                y_train,
                inc,
            )
            # (Collect the PI of the mixture) ----------------------------------
            return self.criterion.ei(
                bma_predictions.squeeze(1), best_f=inc,
                maximize=True)

    @torch.no_grad()
    def get_pi(
            self,
            x_test, inc, x_train=None, y_train=None, minimize=True
    ):
        step = x_train.shape[0]
        x_train, y_train, x_test, inc, = \
            self._preprocess(x_train, y_train, x_test, inc, minimize=minimize)

        for callback in self.callbacks:
            callback.on_acq_start(x_train, y_train, x_test, inc)

        # (Impute related tasks) -----------------------------------------------
        imputed_y = self.imputer(
            x_train=self.related_context.x,
            x_test=x_train.repeat(1, self.num_related, 1),
            y_train=self.related_context.y
        )

        if step < self.min_context_size:
            pi_values = self.warmstart(
                imputed_y,
                self.related_context,
                x_train,
                x_test,
                y_train,
                inc,
                acquisition_fn='pi'
            )

            for callback in self.callbacks:
                callback.on_acq_end_warmstart(x_train, y_train, x_test, inc, pi_values)

            return pi_values


        else:
            predictions = self.mixture_strategy(
                imputed_y,
                self.related_context,
                x_train,
                x_test,
                y_train,
                inc,
            )

            for callback in self.callbacks:
                callback.on_acq_end_mixture(x_train, y_train, x_test, inc, predictions)

            return self.criterion.pi(
                predictions.squeeze(1), best_f=inc,
                maximize=True)

    def warmstart(
            self,
            imputed_y,
            related_context,
            x_train,
            x_test,
            y_train,
            inc,
            acquisition_fn='pi'
    ):

        if self.incumbent_calculation == 'imputation-only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related-only':
            prior_incumbents = related_context.y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation-and-related':
            prior_incumbents = torch.cat([related_context.y, imputed_y, ], dim=0).max(
                dim=0).values
        else:
            raise ValueError(
                f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        # evaluate the query points under the related tasks augmented with the
        # imputed values at the location of observed points under the target task
        prior_logits = self.model(
            (
                torch.cat([
                    related_context.x,
                    x_train.repeat(1, self.num_related, 1),
                    x_test.repeat(1, self.num_related, 1)
                ], dim=0),
                torch.cat([related_context.y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context.x.shape[0] + x_train.shape[0],
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(self.num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        # get the pi at the query points under the related tasks,
        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])
        acq_fn = getattr(self.model.criterion, acquisition_fn)
        acq_related = torch.stack([
            acq_fn(
                prior_logits[:, b, :].squeeze(1),
                best_f=prior_incumbents[b, :].unsqueeze(1),
                maximize=True
            )
            for b in range(self.num_related)
        ], dim=0)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],

        )
        acq_target = acq_fn(
            target_logits.squeeze(1), best_f=inc,
            maximize=True
        )

        self.last_step_predictions = torch.cat([target_logits, prior_logits], dim=0)
        self.past_x_test = x_test

        # here we want to be maximally aggressive from the perspective of the priors,
        # and encourage exploring successful incumbents under the related tasks
        return torch.cat([acq_target.unsqueeze(0), acq_related], dim=0).max(axis=0).values

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

        # TODO the following two forwards can be batched together with
        #  appropriate padding masks. this will save wallclock time
        # (Collect target task logits) --------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )

        # (Collect prior logits) -------------------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        imputation_augmented_prior_logits = self.model(
            (
                torch.cat([
                    # train
                    related_context.x,
                    x_train.repeat(1, self.num_related, 1),

                    # Query
                    x_test.repeat(1, self.num_related, 1),
                ], dim=0),
                torch.cat([related_context.y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context.x.shape[0] + x_train.shape[0],
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(self.num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        if self.model_avg == 'ppd_mixture_eqw':
            query_size = x_test.shape[0]
            logits = torch.concat([
                target_logits[:query_size],
                imputation_augmented_prior_logits[:query_size]
            ], dim=1)
            prediction = logits.mean(dim=1)
            return prediction

        # (Project prior logits into target task) --------------------------
        # Here we take the predicted prior logits of the x_test and need to adjust them
        # according to the error logits. -- which tell us how to shift the distribution
        # (median) and given the shift, how to adjust the probability mass.
        y_error = y_train.repeat(1, self.num_related) - imputed_y
        projected_logits, error_logits = self.error_model.convolve_probs_with_error(
            logits=imputation_augmented_prior_logits,
            x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
            x_test=x_test[:, :, 1:].repeat(1, self.num_related, 1),
            y_error=y_error
        )

        for callback in self.callbacks:
            callback.on_trained_ppds(
                target_logits, imputation_augmented_prior_logits, error_logits, projected_logits,
                imputed_y, y_error,
                x_train, y_train, x_test, inc
            )

        # now the convolved logits describe:
        # x_test, related_context.x, (and if debug=True x_train) in the target task space

        # (Bayesian model averaging) -----------
        # prior_predictions = convolved_logits[:-step]

        # p(y|M_i) = p(y|M_i, D) p(D|M_i) but as logits!
        # predictions = torch.concat([target_logits, prior_predictions], dim=1).to(self.device)

        query = x_test.shape[0]
        predictions = torch.cat(
            [target_logits[:query], projected_logits[:query]], dim=1
        ).to(self.device)

        self.last_step_predictions = predictions
        self.past_x_test = x_test

        if self.model_avg == 'project_eqw':  # equally weighted average
            # here we simply average the predictions over the projected prior with convolved
            # related tasks
            return predictions.mean(dim=1)

        prior_weights, prior_evidence = self.updated_prior_validation(
            x_train=x_train,
            y_train=y_train,
            related_context=related_context,
            imputed_y=imputed_y,
        )

        if debug:
            import matplotlib.pyplot as plt
            df = self.logger.df[self.logger.df['metrics'] == 'prior_weights']
            df.set_index('step')
            df = df[[col for col in df.columns if
                     col.startswith('bma_weight')]]

            df.plot(figsize=(10, 6))
            plt.xlabel('Step')
            plt.ylabel('Weight')
            plt.title('Time Series of Prior PPD BMA Weights')
            plt.grid(True)
            plt.show()

        # given the prior weights, we can now calculate the weighted average of the
        # prior predictions:
        # p(y|.) = \sum_i  p(y|M_i) p(M_i | D)
        prior_prediction = (predictions[:, 1:] * prior_weights.unsqueeze(-1)).sum(dim=1)

        if self.model_avg == 'prior-mixture':
            # here we simply average the predictions over the related tasks
            return prior_prediction

        if self.model_avg == 'bma-decay':
            # In this case we do
            # \alpha(step) p(y|M_{\tau^*}) + (1-\alpha(step)) p_BMA(y|{M_\tau}_{\tau
            # \neq \tau^*})
            # where \tau^* is the target task and \alpha(step) is an exponential decay function
            # that decays with the number of steps.

            alpha = constant_exponential(
                step,
                lambda_=self.kwargs.get('decay_rate', 0.003),
                constant=self.min_context_size
            )

            if debug:
                import numpy as np
                min_context_size = 10  # constant parameter
                lambda_ = 0.003  # decay rate
                steps = np.arange(0, 1001)  # 0-1000 inclusive
                alpha = constant_exponential(steps, lambda_, min_context_size)
                plt.figure(figsize=(10, 6))
                plt.plot(steps, alpha)
                plt.title('Alpha decay over steps')
                plt.xlabel('Step')
                plt.ylabel('Alpha')
                plt.grid(True)
                plt.show()

            return alpha * predictions[:, 0] + (1 - alpha) * prior_prediction

        if self.model_avg == 'bma-cv-target':
            assert x_train.shape[0] >= 10, \
                "BMA with cross-validation requires at least 10 training points."
            target_nll = calc_target_cv_nll(
                x_train.squeeze(1),
                y_train.squeeze(1),
                self.model,
                self.criterion,
                splits=5,
                random_state=42
            ).unsqueeze(0).to(self.device)

            weights = torch.softmax(torch.cat([-target_nll, - prior_evidence]), dim=-1)
            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )
            if debug:
                import matplotlib.pyplot as plt
                df = self.logger.df[self.logger.df['metrics'] == 'prior_weights']
                df.set_index('step')
                df = df[[col for col in df.columns if
                         col.startswith('bma_weight')]]

                df.plot(figsize=(10, 6))
                plt.xlabel('Step')
                plt.ylabel('Weight')
                plt.title('Time Series of Prior PPD BMA Weights')
                plt.grid(True)
                plt.show()

            return (predictions * weights.unsqueeze(-1)).sum(dim=1)

        if self.model_avg == 'bma':
            raise NotImplementedError(
                "BMA model averaging is not implemented yet, "
                "we didn't find a way to calculate the evidence p(H | M_{\\tau^*})."
            )

        if self.model_avg == 'past_surprise':

            # Here we look at the history of the respective model's predictions
            # and find out how each model (target and related) were surprised by the outcome
            config_idx = x_train[-1, :, 0] == self.past_x_test[:, :, 0]
            config = (x_train[-1, :, 2:] == self.past_x_test[:, :, 2:]).all(
                dim=-1)  # ignore fidelity!
            last_prediction = self.last_step_predictions[config_idx.flatten() & config.flatten(), :,
                              :]

            surprise = torch.stack([
                self.criterion(last_prediction[:, b, :].squeeze(1), y_train[-1, :, ])
                for b in range(self.num_related + 1)
            ], dim=0).to(self.device).mean(dim=1)

            self.past_surprises.append(surprise)

            # FIXME: we can adjust the surprise by how wrong the error model was and by how much
            #  we know better how the error looks like for this point now!
            surprise = torch.stack(self.past_surprises, dim=0).mean(dim=0)

            # we will want to use the history of surprises
            weights = torch.softmax(-surprise, dim=-1)
            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )

            return (predictions * weights.unsqueeze(-1)).sum(dim=1)

        if self.model_avg == 'past_suprise_updated_error':
            # we take the best possible prediction, by retrospectively updating the predictions
            # this will improve the prior's projections and we will get a better sense
            # for whether the prior was surprised by the outcome - this will implicitly grant
            # access to the future and in turn will adversely bias
            # against the target task predictions, because it won't be updated
            pass


def constant_exponential(n_target, lambda_=0.001, constant=0):
    effective_n = torch.maximum(torch.tensor(n_target - constant, dtype=torch.float32),
                                torch.tensor(0.0))
    return 1 - torch.exp(-lambda_ * effective_n)
