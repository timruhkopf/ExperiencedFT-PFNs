import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from ifbo.transformer import TransformerModel

from src.model.abstractmodel import AbstractModel
from src.utils.dotdict import DotDict
import pfns4bo

from src.model.utils import general_power_transform
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


class PPFN(AbstractModel):
    __name__ = "pPFN"

    def __init__(self, model, criterion, logger,
                 related_task_data, min_context_size, imputation_mode='median',
                 incumbent_calculation='imputation-only', flippable_related=False,
                 acquisition='pi', model_avg='bma', apply_power_transform = False,
                 device=None, verbose=True):
        """

        :param model:
        :param criterion:
        :param logger:
        :param related_task_data:
        :param min_context_size:
        :param imputation_mode:
        :param incumbent_calculation: Options: ['imputation-only', 'related-only', 'imputation-and-related']
        :param flippable_related:
        :param device:
        :param verbose:
        """
        if "PFNs4BO" in type(model).__name__:
            self.model = model
        else:
            self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model

        self.model.eval()

        # $HOME/anaconda3/envs/ft-pfn-experimental/lib/python3.10/site-packages/pfns4bo/final_models
        # /hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt.gz
        self.error_model = torch.load(pfns4bo.bnn_model, weights_only=False)
        self.error_model.eval()
        self.error_model.to(device)

        self.criterion = criterion if criterion is not None else self.model.criterion

        self.logger = logger
        self.device = device

        self.related_task_data = related_task_data

        self.min_context_size = min_context_size
        self.imputation_mode = imputation_mode
        self.incumbent_calculation = incumbent_calculation
        self.flippable_related = flippable_related
        self.acquisition = acquisition
        self.acquisition_fn = getattr(self.model.criterion, self.acquisition)
        self.model_avg = model_avg
        self.apply_power_transform = apply_power_transform

        self.num_pulling = None
        self.t = 1
        self.use_my_mixture_strategy= True
        self.target_logits = None 

        assert self.acquisition in ['pi', 'ei', 'ucb'], \
            f'Unknown acquisition function: {self.acquisition}. '

        self.verbose = verbose

    @torch.no_grad()
    def get_pi(
            self,
            x_test, inc, x_train=None, y_train=None, minimize=True
    ):

        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask

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

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)

        step = x_train.shape[0]

        related_context_x = related_context_x.to(self.device)
        related_context_y = related_context_y.to(self.device)

        # (Adjust searchspaces) ------------------------------------------------
        # in case the search spaces are supersets of each other, we need to augment the
        # x_train data to match the related context (we need to drop the dim later for the
        # acquisition function to not notice)
        if x_train.shape[-1] < related_context_x.shape[-1]:
            diff = related_context_x.shape[-1] - x_train.shape[-1]
            placeholder = related_context_x[:, :, -diff:].mean(dim=1).mean(dim=0)
            x_train = torch.cat([
                x_train,
                placeholder.repeat(x_train.shape[0], 1)
            ], dim=-1).to(self.device)

            x_test = torch.cat([
                x_test,
                placeholder.repeat(x_test.shape[0], 1)
            ], dim=-1).to(self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        if  "naive" in self.model_avg:
            if "-" in self.model_avg:
                acq_function_name = self.model_avg.split("-")[1]
            else:
                acq_function_name = "pi"
            return self.my_naive_idea(
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
                acq_function_name=acq_function_name
                )

        elif  "pca" in self.model_avg:
            if "-" in self.model_avg:
                acq_function_name = self.model_avg.split("-")[1]
            else:
                acq_function_name = "pi"
            return self.my_pca_idea(
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
                acq_function_name=acq_function_name
                )

        elif "simple" in self.model_avg:
            if "-" in self.model_avg:
                acq_function_name = self.model_avg.split("-")[1]
            else:
                acq_function_name = "pi"
            return self.my_simple_idea(
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
                acq_function_name=acq_function_name
            )

        # (Impute related tasks) -----------------------------------------------
        imputed_y = self.impute(
            related_context_x,
            related_context_y,
            padding_mask,
            x_train
        )
        if self.use_my_mixture_strategy:
            if step < self.min_context_size:
                return self.my_warmstart_pi(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
                )
            else:
                return self.my_mixture_strategy(
                    related_context_x,
                    related_context_y,
                    padding_mask,
                    x_train,
                    x_test,
                    y_train,
                    inc,
                )


        if step < self.min_context_size:
            return self.warmstart_pi(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
            )
        else:
            return self.mixture_strategy(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
            )

    def my_pca_idea(
            self,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
            acq_function_name = 'pi'
    ):
        num_related = related_context_x.shape[1]
        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()

        imputed_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                related_context_y
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
            )


        imputed_train = self.criterion.mean(imputed_logits[:x_train.shape[0], :, :])
        imputed_test = self.criterion.mean(imputed_logits[-x_test.shape[0]:, :, :])


        max_meta_feature_size = 18
        x_train_combined = torch.cat([ x_train, imputed_train.unsqueeze(1) ], dim=-1)
        x_test_combined = torch.cat([ x_test, imputed_test.unsqueeze(1) ], dim=-1)

        if x_train_combined.shape[1] > max_meta_feature_size:
            x_train_np = x_train_combined.cpu().numpy()
            x_test_np = x_test_combined.cpu().numpy()

            pca = PCA(n_components=max_meta_feature_size)
            x_train_pca = pca.fit_transform(x_train_np)
            x_test_pca = pca.transform(x_test_np)

            x_train_combined = torch.tensor(x_train_pca, dtype=x_train.dtype, device=x_train.device)
            x_test_combined = torch.tensor(x_test_pca, dtype=x_test.dtype, device=x_test.device)


        target_logits = self.model(
            (
                torch.cat([x_train_combined, x_test_combined], dim=0),
                y_train
            ),
            single_eval_pos=x_train_combined.shape[0],

        )
        self.target_logits = target_logits

        if acq_function_name == 'ei':
            return self.model.criterion.ei(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )
        else:
            return self.model.criterion.pi(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )

    def my_naive_idea(
            self,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
            acq_function_name='pi'
    ):
        num_related = related_context_x.shape[1]
        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()

        imputed_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                related_context_y
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
            )


        imputed_train = self.criterion.mean(imputed_logits[:x_train.shape[0], :, :])
        imputed_test = self.criterion.mean(imputed_logits[-x_test.shape[0]:, :, :])


        max_meta_feature_size = 18 - x_train.shape[-1]
        if num_related > max_meta_feature_size:
            idx = torch.randperm(num_related)[:max_meta_feature_size]
            imputed_train = imputed_train[:, idx]
            imputed_test = imputed_test[:, idx]

        x_train_combined = torch.cat([ x_train, imputed_train.unsqueeze(1) ], dim=-1)
        x_test_combined = torch.cat([ x_test, imputed_test.unsqueeze(1) ], dim=-1)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train_combined, x_test_combined], dim=0),
                y_train
            ),
            single_eval_pos=x_train_combined.shape[0],

        )
        self.target_logits = target_logits
        if acq_function_name == 'ei':
            return self.model.criterion.ei(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )
        else:
            return self.model.criterion.pi(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )


    def my_simple_idea(
            self,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
            acq_function_name='pi'
    ):
        num_related = related_context_x.shape[1]
        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()

        pfns_max_feature_size = 18
        if num_related > pfns_max_feature_size - 1:
            idx = torch.randperm(num_related)[:pfns_max_feature_size - 1]
            related_context_x = related_context_x[:, idx]
            related_context_y = related_context_y[:, idx]
            padding_mask = padding_mask[idx]
            num_related = pfns_max_feature_size - 1

        imputed_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                related_context_y
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
            )


        imputed_train = self.criterion.mean(imputed_logits[:x_train.shape[0], :, :])
        imputed_test = self.criterion.mean(imputed_logits[-x_test.shape[0]:, :, :])

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],

        )


        target_preds_train = self.criterion.mean(target_logits[:x_train.shape[0], :, :])
        target_preds_test = self.criterion.mean(target_logits[-x_test.shape[0]:, :, :])

        x_train_combined = torch.cat([ target_preds_train.unsqueeze(1), imputed_train.unsqueeze(1) ], dim=-1)
        x_test_combined = torch.cat([ target_preds_test.unsqueeze(1), imputed_test.unsqueeze(1) ], dim=-1)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train_combined, x_test_combined], dim=0),
                y_train
            ),
            single_eval_pos=x_train_combined.shape[0],

        )

        self.target_logits = target_logits

        if acq_function_name == 'ei':
            return self.model.criterion.ei(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )
        else:
            return self.model.criterion.pi(
                target_logits.squeeze(1),
                maximize=True,
                best_f=y_train.max(),
            )


    def my_warmstart_pi(
            self,
            imputed_y,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
    ):
        num_related = related_context_x.shape[1]
        if self.num_pulling is None:
            #print("Warmstarting the pPFN with num_pulling = 10e-3")
            self.num_pulling = torch.zeros(1 + num_related, dtype=torch.int32).to(self.device) + 10e-3
            self.t = 1

        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, imputed_y[:, i].unsqueeze(1)) for i in range(imputed_y.shape[1])]
            imputed_y = torch.cat(transformed_cols, dim=1)
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()

        if self.incumbent_calculation == 'imputation-only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related-only':
            prior_incumbents = related_context_y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation-and-related':
            prior_incumbents = torch.cat([related_context_y, imputed_y, ], dim=0).max(
                dim=0).values
        else:
            raise ValueError(
                f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        # evaluate the query points under the related tasks augmented with the
        # imputed values at the location of observed points under the target task
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            src_key_padding_mask=torch.cat([
                padding_mask,
                torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            ], dim=1)
        )

        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])


        ucb_related = torch.stack([
            self.model.criterion.ucb(
                prior_logits[:, b, :].squeeze(1),
                maximize=True,
                best_f=prior_incumbents[b, :],
                rest_prob=0.99
            )
            for b in range(num_related)
        ], dim=0)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],

        )
        ucb_target = self.model.criterion.ucb(
            target_logits.squeeze(1),
            maximize=True,
            best_f=inc,
            rest_prob=0.99
        )

        max_ucb = torch.cat([ucb_target.unsqueeze(0), ucb_related], dim=0).max(axis=1).values
        arm = max_ucb.argmax()
        return torch.cat([ucb_target.unsqueeze(0), ucb_related], dim=0)[arm]

    def my_mixture_strategy(
            self,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
    ):
        step = x_train.shape[0]
        num_related = related_context_x.shape[1]
        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()
        # (Collect target task logits) --------------------------------
        target_logits = self.model(
            (
                torch.cat([
                    x_train,
                    x_test,
                    x_train  # for BMA: p(D|M)
                ], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )
        target_evidence = target_logits[-step:] # logits for BMA: p(D|M)
        target_evidence = self.criterion(target_evidence, y_train).mean(dim=0)
        target_logits = target_logits[:-step]

        # (Collect prior logits) -------------------------------------------
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_test.repeat(1, num_related, 1),
                    x_train.repeat(1, num_related, 1) # for BMA: p(D|M)
                ], dim=0),
                related_context_y
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
        )


        imputed_y = self.criterion.mean(prior_logits[-step:, :, :])
        print(y_train)
        print(imputed_y)


        # (Collect difference function) ------------------------------------
        # we calcualte the difference function between the related tasks and target task
        # anchored in the imputed values. Since we are only interested in the
        error_logits = self.error_model(
            (
                torch.cat([
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1),
                    x_train.repeat(1, num_related, 1) 
                ], dim=0),
                (y_train.repeat(1, num_related) - imputed_y)
            ),
            single_eval_pos=x_train.shape[0],
            # fixme: this model is not capable of accepting padding masks yet!
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )
        

        # TODO: if the BMA needs point predictors for the logit, we can simply append
        #  the position in the forward, adjust the padding mask length and finally split
        #  the output logits into those requested by the BMA and the acquisition's query points

        # (Project prior logits into target task) --------------------------
        # Here we take the predicted prior logits of the x_test and need to adjust them
        # according to the error logits. -- which tell us how to shift the distribution
        # (median) and given the shift, how to adjust the probability mass.
        # given the prior logits and error logits are differently binned distributions
        # we need to interpret the error bins and adjust probabiltiy mass of the prior logits

        # This projection can be done by computing the fractional overlap of each original bin
        # with each common bin and distributing the original bin’s probability accordingly.
        # now let us move the error logits into the prior logits space
        target_borders = self.criterion.borders
        error_borders = self.error_model.criterion.borders
        print(target_borders, error_borders)
        print("checking error borders")
        aa = F.softmax(error_logits, dim=-1)
        err = self.error_model.criterion.mean(aa[-step:, :, :])
        print(err)

        error_probs = project_probs_to_common_bins_batch(
            F.softmax(error_logits, dim=-1),
            error_borders,
            target_borders
        )

        err = self.criterion.mean(error_probs[-step:, :, :])
        print(err)
        print("done ")


        prior_probs = F.softmax(prior_logits, dim=-1)
        # now we need to convolve the prior_probs with the error probs -------
        T, B, D = prior_probs.shape
        convolved_logits = convolve_probs_with_error(
            prior_probs.view(-1, D),
            error_probs.view(-1, D),
            bin_centers=target_borders[1:]- target_borders[:-1] /2
        ).reshape(T, B, -1)

        conv = self.criterion.mean(convolved_logits[-step:, :, :])
        print(conv)

        # (Bayesian model averaging) -----------
        # logits for BMA: p(D|M)
        prior_evidence = convolved_logits[-step:]  
        prior_evidence = torch.stack([
            self.criterion(prior_evidence[:, b, :].unsqueeze(1), y_train)
            for b in range(B)
        ], dim=0).mean(dim=1).squeeze(1)
        prior_predictions = convolved_logits[:-step]

        # p(y|M_i) = p(y|M_i, D) p(D|M_i) but as logits!
        predictions = torch.concat([target_logits, prior_predictions], dim=1).to(self.device)
        if self.model_avg == 'bma':
            # p(D|M_i)
            evidence = torch.cat([target_evidence, prior_evidence], dim=0).to(self.device)
            unnormalized_posteriors = torch.exp(evidence)
            # p(M_i | D) = p(D|M_i) p(M_i) / [\sum_j p(D|M_j) p(M_j)]
            # here we assume that the prior probabilities are uniform, i.e. p(M_i) = 1 / n_related
               # p(y|M_i)
            posterior_model_probs = unnormalized_posteriors / unnormalized_posteriors.sum(dim=0 )
            # Expand posterior probabilities to match prediction dims for weighting
            weights = posterior_model_probs.unsqueeze(-1)

            # Weighted average of predictive probabilities
            # prediction: p(y|.) = \sum_i  p(y|M_i) p(M_i | D)
            bma_prediction = (predictions * weights).sum(dim=1)

        elif self.model_avg == 'eqw': # equally weighted average
            # here we simply average the predictions over the related tasks
            bma_prediction = predictions.mean(dim=1)


        # (Collect the PI of the mixture) ----------------------------------
        return self.acquisition_fn(
            bma_prediction.squeeze(1), best_f=inc,
            maximize=True)





    def impute(
            self, related_context_x, related_context_y, padding_mask, x_train
    ):
        num_related = related_context_x.shape[1]
        # (IMPUTATION to related tasks) ----------------------------------------
        imputed_logits = self.model(
            (
                torch.cat([related_context_x, x_train.repeat(1, num_related, 1)],
                          dim=0),
                torch.cat([related_context_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
        )

        if self.imputation_mode == 'median':
            imputed_y = self.criterion.median(imputed_logits)
        elif self.imputation_mode == 'mean':
            imputed_y = self.criterion.mean(imputed_logits)
        elif self.imputation_mode == 'sample':
            # Sample indices from the categorical distributions
            # Shape: (T, n_related_tasks, 1)
            probs = imputed_logits.softmax(-1)
            bins = self.criterion.borders

            # to get the sample at the middle of the bins, we can calculate the middle points
            bucket_middle = (bins[:-1] + bins[:-1] + self.criterion.bucket_widths) / 2

            sampled_indices = torch.stack([
                torch.multinomial(probs[:, i, :], 1) for i in range(probs.shape[1])
            ], dim=1)

            # Remove the last dimension for direct indexing
            # Shape: (T, n_related_tasks)
            sampled_indices = sampled_indices.squeeze(-1)

            # Gather the corresponding bin values
            # Shape: (T, n_related_tasks, num_bars) if bins is 2D, else (T, n_related_tasks)
            imputed_y = bucket_middle[sampled_indices]
        else:
            raise ValueError(f"Unknown imputation mode: {self.imputation_mode}")

        return imputed_y

    def warmstart_pi(
            self,
            imputed_y,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
    ):
        num_related = related_context_x.shape[1]
        if self.num_pulling is None:
            #print("Warmstarting the pPFN with num_pulling = 10e-3")
            self.num_pulling = torch.zeros(1 + num_related, dtype=torch.int32).to(self.device) + 10e-3
            self.t = 1

        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, imputed_y[:, i].unsqueeze(1)) for i in range(imputed_y.shape[1])]
            imputed_y = torch.cat(transformed_cols, dim=1)
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()

        if self.incumbent_calculation == 'imputation-only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related-only':
            prior_incumbents = related_context_y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation-and-related':
            prior_incumbents = torch.cat([related_context_y, imputed_y, ], dim=0).max(
                dim=0).values
        else:
            raise ValueError(
                f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        # evaluate the query points under the related tasks augmented with the
        # imputed values at the location of observed points under the target task
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            src_key_padding_mask=torch.cat([
                padding_mask,
                torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            ], dim=1)
        )

        # get the pi at the query points under the related tasks,
        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])

        pi_related = torch.stack([
            self.acquisition_fn(
                prior_logits[:, b, :].squeeze(1),
                best_f=prior_incumbents[b, :],
                maximize=True
            )
            for b in range(num_related)
        ], dim=0)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],

        )
        pi_target = self.acquisition_fn(
            target_logits.squeeze(1), best_f=inc,
            maximize=True
        )

        # max_ucb = torch.cat([pi_target.unsqueeze(0), pi_related], dim=0).max(axis=1).values + ((0.5 * torch.log(torch.tensor(self.t, dtype=torch.float32, device=self.device)) / self.num_pulling) ** 2)
        # arm = max_ucb.argmax()
        # self.t += 1
        # if(arm> 0):
        #     self.num_pulling[arm] += 1
        # return torch.cat([pi_target.unsqueeze(0), pi_related], dim=0)[arm]

        # here we want to be maximally aggressive from the perspective of the priors,
        # and encourage exploring successful incumbents under the related tasks
        return torch.cat([pi_target.unsqueeze(0), pi_related], dim=0).max(axis=0).values

    def mixture_strategy(
            self,
            imputed_y,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
    ):
        step = x_train.shape[0]
        num_related = related_context_x.shape[1]
        if(self.apply_power_transform):
            transformed_cols = [general_power_transform(y_train, imputed_y[:, i].unsqueeze(1)) for i in range(imputed_y.shape[1])]
            imputed_y = torch.cat(transformed_cols, dim=1)
            transformed_cols = [general_power_transform(y_train, related_context_y[:, i].unsqueeze(1)) for i in range(related_context_y.shape[1])]
            related_context_y = torch.cat(transformed_cols, dim=1)
            y_train = general_power_transform(y_train, y_train)
            inc = y_train.max()
        # TODO the following two forwards can be batched together with
        #  appropriate padding masks. this will save wallclock time
        # (Collect target task logits) --------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        target_logits = self.model(
            (
                torch.cat([
                    x_train,
                    x_test,
                    x_train  # for BMA: p(D|M)
                ], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )
        target_evidence = target_logits[-step:] # logits for BMA: p(D|M)
        target_evidence = self.criterion(target_evidence, y_train).mean(dim=0)
        target_logits = target_logits[:-step]

        

        # (Collect prior logits) -------------------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1),
                    x_train.repeat(1, num_related, 1) # for BMA: p(D|M)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            src_key_padding_mask=torch.cat([
                padding_mask,
                torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            ], dim=1)
        )
        
        # (Collect difference function) ------------------------------------
        # we calcualte the difference function between the related tasks and target task
        # anchored in the imputed values. Since we are only interested in the
        error_logits = self.error_model(
            (
                torch.cat([
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1),
                    x_train.repeat(1, num_related, 1) 
                ], dim=0),
                y_train.repeat(1, num_related) - imputed_y
            ),
            single_eval_pos=x_train.shape[0],
            # fixme: this model is not capable of accepting padding masks yet!
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        # TODO: if the BMA needs point predictors for the logit, we can simply append
        #  the position in the forward, adjust the padding mask length and finally split
        #  the output logits into those requested by the BMA and the acquisition's query points

        # (Project prior logits into target task) --------------------------
        # Here we take the predicted prior logits of the x_test and need to adjust them
        # according to the error logits. -- which tell us how to shift the distribution
        # (median) and given the shift, how to adjust the probability mass.
        # given the prior logits and error logits are differently binned distributions
        # we need to interpret the error bins and adjust probabiltiy mass of the prior logits

        # This projection can be done by computing the fractional overlap of each original bin
        # with each common bin and distributing the original bin’s probability accordingly.
        # now let us move the error logits into the prior logits space
        target_borders = self.criterion.borders
        error_borders = self.error_model.criterion.borders
        

        error_probs = project_probs_to_common_bins_batch(
            F.softmax(error_logits, dim=-1),
            error_borders,
            target_borders,
        )
        prior_probs = F.softmax(prior_logits, dim=-1)

        # now we need to convolve the prior_probs with the error probs -------
        T, B, D = prior_probs.shape
        convolved_logits = convolve_probs_with_error(
            prior_probs.view(-1, D),
            error_probs.view(-1, D),
            bin_centers=target_borders[1:]- target_borders[:-1] /2
        ).reshape(T, B, -1)


        # (Bayesian model averaging) -----------
        # logits for BMA: p(D|M)
        prior_evidence = convolved_logits[-step:]  
        prior_evidence = torch.stack([
            self.criterion(prior_evidence[:, b, :].unsqueeze(1), y_train)
            for b in range(B)
        ], dim=0).mean(dim=1).squeeze(1)
        prior_predictions = convolved_logits[:-step]

        # p(y|M_i) = p(y|M_i, D) p(D|M_i) but as logits!
        predictions = torch.concat([target_logits, prior_predictions], dim=1).to(self.device)
        if self.model_avg == 'bma':
            # p(D|M_i)
            evidence = torch.cat([target_evidence, prior_evidence], dim=0).to(self.device)
            unnormalized_posteriors = torch.exp(evidence)
            # p(M_i | D) = p(D|M_i) p(M_i) / [\sum_j p(D|M_j) p(M_j)]
            # here we assume that the prior probabilities are uniform, i.e. p(M_i) = 1 / n_related
               # p(y|M_i)
            posterior_model_probs = unnormalized_posteriors / unnormalized_posteriors.sum(dim=0 )
            # Expand posterior probabilities to match prediction dims for weighting
            weights = posterior_model_probs.unsqueeze(-1)

            # Weighted average of predictive probabilities
            # prediction: p(y|.) = \sum_i  p(y|M_i) p(M_i | D)
            bma_prediction = (predictions * weights).sum(dim=1)

        elif self.model_avg == 'eqw': # equally weighted average
            # here we simply average the predictions over the related tasks
            bma_prediction = predictions.mean(dim=1)


        # (Collect the PI of the mixture) ----------------------------------
        return self.acquisition_fn(
            bma_prediction.squeeze(1), best_f=inc,
            maximize=True)

def convolve_probs_with_error(probs, error_probs, bin_centers):
    batch_size, length = probs.shape
    kernel_size = error_probs.shape[1]

    # Original input shape: (batch_size, 1, length)
    p_orig_t = probs.view(batch_size, 1, length)

    # Flip kernels for convolution
    p_shift_flipped = torch.flip(error_probs, dims=[1]).view(batch_size, 1, kernel_size)

    # Now, merge batch into channels dimension by transposing:
    # Input: (batch_size, 1, length) -> (1, batch_size, length)
    p_orig_t_merged = p_orig_t.permute(1, 0, 2)  # (1, batch_size, length)

    # Weight already has shape (batch_size, 1, kernel_size)
    # To match input's channels, reshape kernels as (batch_size, 1, kernel_size)
    # Perform conv1d with groups = batch_size
    p_convolved = F.conv1d(
        p_orig_t_merged,  # input channels == batch_size
        p_shift_flipped,
        # weight shape must be (out_channels, in_channels/groups, kernel_size); here out_channels=batch_size, in_channels/groups=1
        padding=kernel_size - 1,
        groups=batch_size
    )

    # p_convolved shape: (1, batch_size, output_length)
    # reshape back to (batch_size, output_length)
    p_convolved = p_convolved.permute(1, 0, 2).view(batch_size, -1)

    # Normalize per batch
    p_convolved /= p_convolved.sum(dim=1, keepdim=True)

    convolved_bin_centers = torch.arange(bin_centers[0] + bin_centers[0],
                                         bin_centers[-1] + bin_centers[-1] + 1,
                                         dtype=torch.float32)

    original_len = bin_centers.shape[0]
    full_len = p_convolved.shape[1]  # Length after convolution (should be 2*original_len - 1)

    # Calculate start and end indices for cropping (center-crop)
    start = (full_len - original_len) // 2
    end = start + original_len

    # Crop convolved output
    p_convolved_cropped = p_convolved[:, start:end]

    # now adjust for the cropping to get propper distribution again:
    missing_prob_right = p_convolved[:, end:].sum(dim=1)
    missing_prob_left = p_convolved_cropped[:, :start].sum(dim=1)
    p_convolved_cropped[:, start] += missing_prob_left
    p_convolved_cropped[:, -1] += missing_prob_right

    eps = 1e-12
    logits_convolved = torch.log(p_convolved_cropped.clamp(min=eps))

    return logits_convolved


# FIXME: double check the implementation of the logit projection
def project_probs_to_common_bins_batch(orig_probs, orig_bounds, target_bounds):
    """
    Project batched probability distributions defined on orig_bounds to target_bounds 
    by fractional overlap of bins in PyTorch.

    Args:
        orig_probs   : Tensor of shape (..., N) -- probabilities over original bins.
        orig_bounds  : 1D tensor of length N+1 -- bin edges of original distribution.
        target_bounds: 1D tensor of length M+1 -- bin edges of target distribution.

    Returns:
        projected_probs : Tensor of shape (..., M) -- probabilities projected onto target bins.
    """
    # orig_probs: (..., N)
    orig_shape = orig_probs.shape
    N = orig_shape[-1]
    M = target_bounds.shape[0] - 1

    # Expand bins for vectorized overlap calculation:
    # orig lefts and rights shape: (N, 1)
    orig_lefts = orig_bounds[:-1].unsqueeze(1)  # (N,1)
    orig_rights = orig_bounds[1:].unsqueeze(1)  # (N,1)
    # target lefts and rights shape: (1, M)
    target_lefts = target_bounds[:-1].unsqueeze(0)  # (1, M)
    target_rights = target_bounds[1:].unsqueeze(0)  # (1, M)

    # Calculate overlaps (N x M)
    overlaps = torch.clamp(torch.min(orig_rights, target_rights) - torch.max(orig_lefts, target_lefts), min=0.0)  # (N,M)
    orig_widths = (orig_rights - orig_lefts)  # (N,1)
    fractions = overlaps / orig_widths  # (N,M)

    # Move orig_probs last dim (N) to front to do batch matmul:
    # orig_probs reshaped to (-1, N)
    orig_probs_flat = orig_probs.reshape(-1, N)  # (B, N)

    # Multiply: (B, N) @ (N, M) => (B, M)
    projected_flat = torch.matmul(orig_probs_flat, fractions)  # (B, M)

    # Normalize so projected probabilities sum to 1 (for each batch)
    #projected_flat /= projected_flat.sum(dim=1, keepdim=True)

    # Reshape back to original batch dims + M
    projected_shape = orig_shape[:-1] + (M,)
    projected_probs = projected_flat.reshape(projected_shape)

    return projected_probs

