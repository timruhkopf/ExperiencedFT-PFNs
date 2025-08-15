import torch


class Counterfactor:
    def __init__(
            self,
            counterfit: str = 'median', num_related: int = 5,
            n_mc: int = 1,  # number of Monte Carlo samples for counterfeiting
    ):
        self.counterfit = counterfit  # 'median' or 'mc'
        self.n_mc = n_mc  # number of Monte Carlo samples for counterfeiting

    def __post_init__(self, parent_model, model, criterion, error_model, related_context, logger,
                                                                     device):
        self.parent_model = parent_model
        self.model = model
        self.criterion = criterion
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.logger = logger
        self.device = device
        self.error_model = error_model  # the error model to project the prior data into the target task space


    def __call__(self, x_train, y_train, y_error, related_context, imputed_y) -> torch.Tensor:
        # Let us collect the counterfactual data:
        # the prior data is projected into the target task space by the
        # learned error model.
        # Notice, that the prior data is observed and therefore has dirac mass
        step = x_train.shape[0]

        counterfactural_logits, error_logits, bardist = self.error_model.dirac_forward(
            x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
            dirac_x=related_context.x[:, :, 1:],
            dirac_y=related_context.y,
            y_error=y_error,
            reverse=True
        )

        # Now we collect the y values for the counterfactual data.
        if self.counterfit == 'median' and self.n_mc ==1:
            counterfactual_y = torch.stack([
                bardist.median(counterfactural_logits[:, b, :].squeeze(1))
                for b in range(self.num_related)
            ], dim=1).to(self.device)

            related_x = related_context.x
            query = x_train.repeat(1, self.num_related, 1)
            n_mc = 1  # for compatability purposes

        elif self.counterfit == 'mc':
            from model.components.imputor import sample_logits


            mc_counterfactural_y = []
            for b in range(self.num_related):
                mc_counterfactural_y.append(
                    sample_logits(
                        counterfactural_logits[:, b, :].squeeze(1),
                        self.n_mc,
                        bardist.borders
                    )
                )

            mc_counterfactural_y = torch.stack(mc_counterfactural_y, dim=1).to(self.device)
            counterfactual_y = mc_counterfactural_y.reshape(self.num_related * self.n_mc, -1).T

            related_x = related_context.x.repeat(1, self.n_mc, 1)
            query = x_train.repeat(1, self.num_related * self.n_mc, 1)


        else:
            raise ValueError(f"Unknown counterfiting mode: {self.counterfit}, n_mc {self.n_mc}.")

        # Get the logits for the target task data under the counterfactual prior PPD
        prior_counterfactual_logits = self.model(
            (
                torch.cat(
                    [
                        related_x,
                        query
                    ]
                ),
                counterfactual_y
            ),
            single_eval_pos=related_context.x.shape[0],
        )

        # finally we evaluate the prior counterfactual against the
        # observed y_train values, telling us how well the projected prior data
        # would explain the observed data. This gives us the relative weight of each prior
        prior_evidence = torch.stack([
            self.criterion(prior_counterfactual_logits[:, b, :].squeeze(1), y_train)
            for b in range(self.num_related * self.n_mc)
        ], dim=0).to(self.device).mean(dim=1)

        prior_evidence = prior_evidence.reshape(self.num_related, self.n_mc, -1).mean(dim=1)
        prior_evidence = prior_evidence.squeeze(1)  # (n_related,)

        prior_weights = torch.softmax(-prior_evidence, dim=-1)
        self.logger.log(
            {'metrics': 'prior_weights', 'step': step,
             **{f'prior_weight_{i}': w.item()
                for i, w in enumerate(prior_weights)}},
        )

        return prior_weights, prior_evidence
