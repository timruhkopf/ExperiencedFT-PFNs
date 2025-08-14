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
        imputed_y = self.parent_model.interim_results['imputed_y']

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
        prior_logits = self.model(
            (
                torch.cat([
                    self.related_context.x,
                    x_train.repeat(1, self.num_related, 1),
                    x_test.repeat(1, self.num_related, 1)
                ], dim=0),
                torch.cat([self.related_context.y, imputed_y, ], dim=0)
            ),
            single_eval_pos=self.related_context.x.shape[0] + x_train.shape[0],
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

        self.parent_model.interim_results.update(dict(
            last_step_predictions=torch.cat([target_logits, prior_logits], dim=1),
            past_x_test=x_test
        ))

        # here we want to be maximally aggressive from the perspective of the priors,
        # and encourage exploring successful incumbents under the related tasks
        return torch.cat([acq_target.unsqueeze(0), acq_related], dim=0).max(axis=0).values