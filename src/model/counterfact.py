import torch
from torch import nn


class Counterfactor:
    def __init__(self, model: nn.Module, device: torch.device,
                 error_model, criterion: nn.Module, logger,
                 counterfit: str = 'median', num_related: int = 5,
                 n_mc: int = 10,  # number of Monte Carlo samples for counterfitting
                 ):

        self.model = model
        self.criterion = criterion

        self.error_model = error_model
        self.num_related = num_related

        self.counterfit = counterfit  # 'median' or 'mc'
        self.n_mc = n_mc  # number of Monte Carlo samples for counterfitting

        self.logger = logger
        self.device = device

    def __call__(self, x_train, y_train, related_context, imputed_y) -> torch.Tensor:
        # Let us collect the counterfactual data:
        # the prior data is projected into the target task space by the
        # learned error model.
        # Notice, that the prior data is observed and therefore has dirac mass
        step = x_train.shape[0]

        counterfactural_logits, error_logits = self.error_model.dirac_forward(
            x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
            dirac_x=related_context.x[:, :, 1:],
            dirac_y=related_context.y,
            y_error=y_train.repeat(1, self.num_related) - imputed_y,
            reverse=True
        )

        # Now we collect the y values for the counterfactual data.
        if self.counterfit == 'median':
            counterfactual_y = torch.stack([
                self.criterion.median(counterfactural_logits[:, b, :].squeeze(1))
                for b in range(self.num_related)
            ], dim=1).to(self.device)

            related_x = related_context.x
            query = x_train.repeat(1, self.num_related, 1)
            n_mc = 1  # for compatability purposes

        elif self.counterfit == 'mc':
            from src.model.imputor import sample_logits

            n_mc = 3
            mc_counterfactural_y = []
            for b in range(self.num_related):
                mc_counterfactural_y.append(
                    sample_logits(
                        counterfactural_logits[:, 0, :].squeeze(1),
                        n_mc,
                        self.criterion.borders
                    )
                )

            mc_counterfactural_y = torch.stack(mc_counterfactural_y, dim=1).to(self.device)
            counterfactual_y = mc_counterfactural_y.reshape(self.num_related * n_mc, -1).T

            related_x = related_context.x.repeat(1, n_mc, 1)
            query = x_train.repeat(1, self.num_related * n_mc, 1)

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
            for b in range(self.num_related * n_mc)
        ], dim=0).to(self.device).mean(dim=1)

        prior_evidence = prior_evidence.reshape(self.num_related, n_mc, -1).mean(dim=1)
        prior_evidence = prior_evidence.squeeze(1)  # (n_related,)

        prior_weights = torch.softmax(-prior_evidence, dim=-1)
        self.logger.log(
            {'metrics': 'prior_weights', 'step': step,
             **{f'prior_weight_{i}': w.item()
                for i, w in enumerate(prior_weights)}},
        )

        return prior_weights, prior_evidence
