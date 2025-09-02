from copy import deepcopy
from functools import partial

import torch

from model.initial_design.abstract_intial_design import AbstractInitialDesign


class MaxPIInitialDesign(AbstractInitialDesign):
    def __init__(self, incumbent_calculation='imputation-only', **kwargs):
        self.incumbent_calculation = incumbent_calculation
        super().__init__(**kwargs)

    def __call__(
            self,
            x_train,
            x_test,
            y_train,
            inc,
            acquisition_fn='pi'
    ) -> torch.Tensor:
        """
        Here, we calculate the Maximum Probability of Improvement (MaxPI) acquisition function
        based on the imputed values of the related tasks and the target task data.

        This is an optimistic acquisition function that encourages exploration
        of the search space by selecting points that are expected to yield the highest
        improvement over the current best known solution (incumbent) based on
        known solutions in the past and the current advancement on the target task
        via imputations

        :param x_train:
        :param x_test:
        :param y_train:
        :param inc:
        :param acquisition_fn:
        :return: Acquisition values
        """
        imputed_y = self.parent_model.interim_results['imputed_y'].to(self.device)

        if self.incumbent_calculation == 'imputation-only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related-only':
            prior_incumbents = self.related_context.y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation-and-related':
            prior_incumbents = torch.cat([self.related_context.y, imputed_y, ], dim=0).max(
                dim=0).values
        else:
            raise ValueError(
                f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        # evaluate the query points under the related tasks augmented with the
        # imputed values at the location of observed points under the target task
        imputation_augmented_prior_model = partial(
            self.get_imputation_augmented_prior,
            x_train=x_train,
            imputed_y=imputed_y
        )
        imp_aug_prior_logits = imputation_augmented_prior_model(x_test=x_test)

        # get the pi at the query points under the related tasks,
        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])
        acq_fn = getattr(self.model.criterion, acquisition_fn)
        acq_related = torch.stack([
            acq_fn(
                imp_aug_prior_logits[:, b, :].squeeze(1),
                best_f=prior_incumbents[b, :].unsqueeze(1),
                maximize=True
            )
            for b in range(self.num_related)
        ], dim=0)

        # get the pi under the target task

        target_model = partial(
            self.get_target_model,
            x_train=x_train, y_train=y_train
        )
        target_logits = target_model(x_test=x_test)

        acq_target = acq_fn(
            target_logits.squeeze(1), best_f=inc,
            maximize=True
        )

        self.parent_model.interim_results.update(dict(
            last_step_predictions=torch.cat([target_logits, imp_aug_prior_logits], dim=1).cpu(),
            past_x_test=x_test.cpu()
        ))

        # here we want to be maximally aggressive from the perspective of the priors,
        # and encourage exploring successful incumbents under the related tasks
        self.suggestions = torch.cat([acq_target.unsqueeze(0), acq_related], dim=0)
        return self.suggestions.max(axis=0).values


class RepeatedMaxPIInitialDesign(MaxPIInitialDesign):
    def __init__(self, repetitions=5, **kwargs):
        self.repetitions = repetitions
        super().__init__(**kwargs)
        self.cashed_suggestion = None
        self.last_x_test = None
        self.counter = 0

    def __call__(
            self,
            x_train,
            x_test,
            y_train,
            inc,
            acquisition_fn='pi'
    ) -> torch.Tensor:
        """
        Here, we choose a configuration based on each prior and repeat it irrespective of the
        change in the acquisition function values for the number of repetitions, before we
        move to the next configuration sampled from the next prior
        :param x_train:
        :param x_test:
        :param y_train:
        :param inc:
        :param acquisition_fn:
        :return:
        """

        if self.counter % self.repetitions == 0:
            pi = super().__call__(
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                inc=inc,
                acquisition_fn=acquisition_fn
            )
            self.cashed_suggestion = deepcopy(self.suggestions)

            # now find the new test config, that we want to pursue for multiple steps
            # once we completed the repetitions, we move to the next prior
            prior_idx = self.counter // self.repetitions % (self.num_related + 1)

            # start with the priors first!
            new_config_idx = self.cashed_suggestion[-(prior_idx+1)].argmax()
            self.new_config = x_test[new_config_idx, :].unsqueeze(1)

        self.counter += 1

        # suggestion = ((x_test[:, :, 0] == self.new_config[:, :, 0]).flatten()  \
        #               * (x_test[:, :, 2:] == self.new_config[:, :, 2:]).flatten())
        suggestion = (x_test[:, :, 2:] == self.new_config[:, :, 2:]).flatten()

        # print(self.new_config, x_test[suggestion])
        return suggestion.float()
