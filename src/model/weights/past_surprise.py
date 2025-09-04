import torch

from ifBO_icml2024.src.PFNs4HPO.pfns4hpo.bar_distribution import BarDistribution
from src.model.components.ema_filter import ema_conv_causal
from src.model.probability_conv.convolver import DistributionConvolver
from src.model.weights.abstract_weights import AbstractWeights

import warnings
import logging

log = logging.getLogger(__name__)


def override_call_decorator(func):
    def wrapper(x_train, x_test, *args, **kwargs):
        n_x_test = x_test.shape[0]

        # we must consider, that we ask for the next fidelity of seen configurations
        train = x_train.clone()
        min_fidelity = train[:, :, 1].min()
        train[:, :, 1] += min_fidelity

        # we need to also consider, that in x_test we ask unseen configurations
        test = x_test.clone()
        test[:, :, 1] = min_fidelity

        # METHOD OVERRIDE!
        result_logits = func(
            x_train=x_train,
            x_test=torch.cat([x_test, train, test], dim=0),
            *args, **kwargs
        )
        test_logits = result_logits[:n_x_test, :, :]
        lookahead_logits = result_logits[n_x_test:, :, :]

        # gain access to the parent model
        self = getattr(func, '__self__', None)
        self.parent_model.interim_results.update({
            f'last-{func.__name__}-lookahead': lookahead_logits.cpu(),
            f'x_lookahead': torch.cat([train[:, 0:1, :], test[:, 0:1, :]], dim=0).cpu()

        })

        return test_logits

    return wrapper


class PastSurpriseWeights(AbstractWeights):

    def __init__(
            self,
            alpha=0.05,
            bias_correction=True,
            truncate=100
    ):
        self.ema_kwargs = {
            'alpha': alpha,
            'bias_correction': bias_correction,
            'truncate': truncate
        }

    def __post_init__(self, related_context, device, logger, model, parent_model):
        super().__post_init__(
            related_context=related_context,
            device=device,
            logger=logger,
            model=model,
            parent_model=parent_model
        )

        # for efficiency reasons, we decorate the model call to compute the one step lookahead
        # logits, so we can have a check what we believed about y for the next step.
        self.parent_model.strategy.get_target_model = override_call_decorator(
            self.parent_model.strategy.get_target_model
        )
        self.parent_model.strategy.get_imputation_augmented_prior = override_call_decorator(
            self.parent_model.strategy.get_imputation_augmented_prior
        )
        if hasattr(self.parent_model.strategy, 'get_error_model'):
            self.parent_model.strategy.get_error_model = override_call_decorator(
                self.parent_model.strategy.get_error_model
            )

        if hasattr(self.parent_model.strategy, 'get_prior_augmented_target_model'):
            self.parent_model.strategy.get_prior_augmented_target_model = override_call_decorator(
                self.parent_model.strategy.get_prior_augmented_target_model
            )

        self.parent_model.interim_results.update({
            'surprise_logits': [],
            'surprise_x': [],
            'surprises_nll': [],
            'last_step_logits': None,
            'past_x_test': None,

        })

    @staticmethod
    def find_extra_row_index(A, B):
        """
        Find the row not present in B, assuming A has exactly one extra row,
        whose location is unknown.

        :example:
            i=3
            A=torch.randn(10,3)
            B = torch.cat([A.clone()[:i, :], torch.ones(1, 3), A.clone()[i:, :]])
            assert i == find_extra_row_index(B,A)
        :param A:
        :param B:
        :return:
        """
        a = set(tuple(v) for v in A.tolist())
        b = set(tuple(v) for v in B.tolist())
        new_idx = torch.tensor(list(b.difference(a))).to(A.device)

        return torch.where((B == new_idx).all(dim=-1).flatten())[0].item()

    def __call__(self, x_train, x_test, y_train, inc, recompute=False, *args,
                 **kwargs) -> torch.Tensor:
        """
        Here we look at the history of the respective model's predictions
        and find out how each model (target and related) were surprised by the outcome
        """
        if x_train.shape[0] == 1:
            return torch.ones(self.num_related + 1).to(self.device) / (self.num_related + 1)

        interim_results = self.parent_model.interim_results
        target_lookahead = interim_results['last-get_target_model-lookahead'].to(self.device)


        new_x = self.find_extra_row_index(
            self.parent_model.interim_results['last-x_train'][:, 0, :].to(self.device),
            x_train[:, 0, :],
        )

        x_lookahead = interim_results['x_lookahead'].to(self.device)

        mask = (x_train[new_x, 0,  2:] == x_lookahead[:, 0, 2:]).all(dim=-1).flatten()
        max_fidelity = x_lookahead[:, 0, 1][mask].max()
        idx = mask & (x_lookahead[:, 0, 1] == max_fidelity)

        if hasattr(self.parent_model.strategy, 'get_error_model'):
            imp_aug_lookahead = interim_results['last-get_imputation_augmented_prior-lookahead'].to(self.device)
            error_logits = interim_results['last-get_error_model-lookahead'].to(self.device)

            borders = self.parent_model.strategy.err_model.criterion.borders
            new_error_logits = torch.where(borders >= -1.)[0].min()
            new_error_logits_end = torch.where(borders <= 1.)[0].max()
            error_logits = error_logits[:, :, new_error_logits:new_error_logits_end]
            error_borders = borders[
                new_error_logits:new_error_logits_end + 1]
            new_error_criterion = BarDistribution(borders=error_borders)


            try:
                projected_logits, projected_criterion = \
                    DistributionConvolver().to(self.device).convolve(
                        A_logits=imp_aug_lookahead[idx, :, :],
                        borders_A=self.model.criterion.borders,
                        B_logits=error_logits[idx, :, :],
                        borders_B=error_borders,
                        reverse=False,  # we convolve the error model with the prior
                        target_borders=self.model.criterion.borders,
                        padding=None  # no padding needed here
                    )

                # import torch
                # import matplotlib.pyplot as plt
                # import seaborn as sns
                #
                # batch_size = imp_aug_lookahead.shape[1]  # 2 (batch dimension)
                # logit_dim = imp_aug_lookahead.shape[2]  # 1000 (logits dimension)
                #
                # fig, axes = plt.subplots(1, batch_size, figsize=(12, 5), squeeze=False)
                #
                # # Apply softmax function along last dimension (dim=2)
                # imp_aug_soft = torch.softmax(imp_aug_lookahead, dim=2)
                # error_soft = torch.softmax(error_logits, dim=2)
                # proj_soft = torch.softmax(projected_logits, dim=2)
                #
                # err_med = self.parent_model.strategy.err_model.criterion.median(error_logits)
                # aug_med = self.model.criterion.median(imp_aug_lookahead)
                # proj_med =self.model.criterion.median(projected_logits)
                #
                # for b in range(batch_size):
                #     ax = axes[0, b]
                #     # Select the batch distributions (1 x batch x logits)
                #     imp_aug_probs = imp_aug_soft[0, b, :].cpu().numpy()
                #     error_probs = error_soft[0, b, :].cpu().numpy()
                #     proj_probs = proj_soft[0, b, :].cpu().numpy()
                #
                #     # Optional: Use borders as bin edges for histogram
                #     borders = self.model.criterion.borders.cpu().numpy()  # shape: (1001,)
                #
                #     # Plot distribution curves as KDE or histograms
                #     sns.lineplot(x=borders[:-1], y=imp_aug_probs, label='imp_aug', ax=ax, color='blue')
                #     sns.lineplot(x=self.parent_model.strategy.err_model.criterion.borders[:-1],
                #                  y=error_probs, label='error', ax=ax, color='red')
                #     sns.lineplot(x=borders[:-1], y=proj_probs, label='projected', ax=ax, color='green')
                #
                #     ax.set_title(f'Batch {b}')
                #     ax.set_xlabel('Logits bins')
                #     ax.set_ylabel('Probability')
                #     ax.legend()
                #     # ax.set_xlim(0, 1)
                #
                # plt.tight_layout()
                # plt.show()

                last_logits = torch.cat(
                    [target_lookahead[idx, :, :], projected_logits], dim=1
                ).to(self.device)
                interim_results['surprise_logits'].append(last_logits.cpu())
                interim_results['surprise_x'].append(x_train[new_x].cpu().squeeze())

                logits = torch.stack(interim_results['surprise_logits'], dim=0).to(
                    self.device).squeeze()

                probs = torch.softmax(logits, dim=-1)

                samples = self.model.criterion.median(logits)

                surprise = torch.stack([
                    self.model.criterion(last_logits[:, b, :].squeeze(1), y_train[new_x, :,])
                    for b in range(self.num_related + 1)
                ], dim=0).to(self.device).mean(dim=1)

                interim_results['surprises_nll'].append(surprise.cpu())

                # FIXME: we can adjust the surprise by how wrong the error model was and by how much
                #  we know better how the error looks like for this point now!
            except Exception as e:
                # this usually is a rare conv error
                log.error(f"Error while computing past surprises: {e}")
                warnings.warn(
                    "Error while computing past surprises",
                    UserWarning
                )
        else:
            imp_aug_target_lookahead = interim_results[
                'last-get_prior_augmented_target_model-lookahead'].to(self.device)

            last_logits = torch.cat(
                [target_lookahead[idx, :, :], imp_aug_target_lookahead[idx, :, :]], dim=1
            ).to(self.device)
            if len(last_logits) > 0:
                interim_results['surprise_logits'].append(last_logits.cpu())
                interim_results['surprise_x'].append(x_train[new_x].cpu().squeeze())

                # logits = torch.stack(interim_results['surprise_logits'], dim=0).to(
                #     self.device).squeeze()
                #
                # probs = torch.softmax(logits, dim=-1)
                #
                # samples = self.model.criterion.median(logits)

                surprise = torch.stack([
                    self.model.criterion(last_logits[:, b, :].squeeze(1), y_train[new_x, :,])
                    for b in range(self.num_related + 1)
                ], dim=0).to(self.device).mean(dim=1)

                interim_results['surprises_nll'].append(surprise.cpu())
            else:
                warnings.warn(
                    "No new lookahead logits were found.",
                    UserWarning
                )
        if any([len(t)>0 for t in interim_results['surprises_nll'] ]):
            surprises = torch.stack(interim_results['surprises_nll'], dim=0).to(
                self.device)
        else:
            surprises = -torch.ones(1, self.num_related + 1).to(self.device)

        surprises = ema_conv_causal(surprises, **self.ema_kwargs)

        # we will want to use the history of surprises
        weights = torch.softmax(-surprises[-1], dim=-1).to(self.device)

        return weights

    def plot(self, ax=None, show=True):
        """
        Plot the past surprises.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        past_surprises = torch.stack(self.parent_model.interim_results['surprises_nll'],
                                     dim=0).to(self.device)
        surprises = ema_conv_causal(past_surprises, **self.ema_kwargs)
        weights = torch.softmax(-surprises, dim=-1).cpu().numpy()
        ax.plot(weights)
        ax.set_title('surprises_nll')
        ax.set_xlabel('Step')
        ax.set_ylabel('weight')

        ax.legend([f'Model {i}' for i in range(self.num_related + 1)])

        if show:
            plt.show()
        return ax
