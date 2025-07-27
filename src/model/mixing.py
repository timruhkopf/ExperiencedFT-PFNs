import torch
import numpy as np
from src.model.calc_reliability import calc_target_cv_nll, calc_imputed_linalg_reliability

class myCVMixtureStrategy:
    def __init__(self, model, criterion, related_task_data=None, min_num_samples=1, logger=None, multi_fidelity=False, transformation_type=None):
        """This class is a new variant of the MixtureStrategy that consider
        the reliability of related scores in conjunction with cross-valdiated nll scores
        of the target data."""
        self.model = model
        self.criterion = criterion
        self.related_task_data = related_task_data
        self.min_num_samples = min_num_samples
        self.logger = logger
        self.multi_fidelity = multi_fidelity
        self.transformation_type = transformation_type
        self.num_related = None
        self.num_pullings = None
        self.t = 0
        self.return_average_loss = True


    def __call__(self, x_train, y_train, pi_target, pi_related, minimize):
        # fixme what do we need to do with the minimize flag?

        device = x_train.device

        if x_train.shape[0] > self.min_num_samples:
            target_nll = calc_target_cv_nll(
                x_train,
                y_train,
                self.model,
                self.criterion,
                splits=min(5, x_train.shape[0]),
                random_state=42,
                start_feature_indx=2 if self.multi_fidelity else 0,
            ).unsqueeze(0).to(device)
            related_nll = calc_imputed_linalg_reliability(
                self.model, x_train, y_train,
                self.related_task_data,
                self.criterion,
                degree_fn=lambda x, y: max(x[:, 0, 1].unique().shape[0] - 3, 0) if self.multi_fidelity else 1,    
                multi_fidelity = self.multi_fidelity,
                transformation_type=self.transformation_type,
                return_average_loss=self.return_average_loss,
                # avoid multicollinearity if all have same fidelity. grow polynomial features based on the fidelity availability
            )
            related_nll = related_nll.to(device)
        else:
            # uniform scores
            num_related = pi_related.shape[0]
            target_nll = torch.zeros(1, device=device)
            related_nll = torch.zeros(num_related, device=device)

        # weigh the target and related scores by the reliability
        reliability = torch.nn.functional.softmax(torch.concat([-target_nll, -related_nll], dim=0), dim=0)

        t = x_train.shape[0]
        reliability[1:] = reliability[1:] / t

        pi_values = torch.concat([pi_target.unsqueeze(0), pi_related], dim=0)
        weighted_pi = (pi_values * reliability.unsqueeze(1)).sum(dim=0, keepdim=True)
        return weighted_pi, reliability


class CVMixtureStrategy:
    def __init__(self, model, criterion, related_task_data=None, min_num_samples=1, logger=None, multi_fidelity=False, transformation_type=None):
        """This class is a new variant of the MixtureStrategy that consider
        the reliability of related scores in conjunction with cross-valdiated nll scores
        of the target data."""
        self.model = model
        self.criterion = criterion
        self.related_task_data = related_task_data
        self.min_num_samples = min_num_samples
        self.logger = logger
        self.multi_fidelity = multi_fidelity
        self.transformation_type = transformation_type

    def __call__(self, x_train, y_train, pi_target, pi_related, minimize):
        # fixme what do we need to do with the minimize flag?

        device = x_train.device
        if x_train.shape[0] > self.min_num_samples:
            target_nll = calc_target_cv_nll(
                x_train,
                y_train,
                self.model,
                self.criterion,
                splits=min(5, x_train.shape[0]),
                random_state=42,
                start_feature_indx=2 if self.multi_fidelity else 0,
            ).unsqueeze(0).to(device)
            related_nll = calc_imputed_linalg_reliability(
                self.model, x_train, y_train,
                self.related_task_data,
                self.criterion,
                degree_fn=lambda x, y: max(x[:, 0, 1].unique().shape[0] - 3, 0) if self.multi_fidelity else 1,    
                multi_fidelity=self.multi_fidelity,
                transformation_type=self.transformation_type,
                # avoid multicollinearity if all have same fidelity. grow polynomial features based on the fidelity availability
            )
            
            related_nll = related_nll.to(device)

            #print(f"Target NLL: {target_nll}, Related NLL: {related_nll}")
        else:
            # uniform scores
            num_related = pi_related.shape[0]
            target_nll = torch.zeros(1, device=device)
            related_nll = torch.zeros(num_related, device=device)


        # filtering out low reliability scores (e.g., unrelated tasks  )

        # weigh the target and related scores by the reliability
        reliability = torch.nn.functional.softmax(
            torch.concat([-target_nll, -related_nll], dim=0), dim=0
        )


        #print(f"Reliability: {reliability}")
        #reliability[1:] = 0


        if self.logger is not None:
            self.logger.add_scalar(
                'reliability', x_train.shape[0], *reliability.detach().tolist(),
            )

        pi_values = torch.concat([pi_target.unsqueeze(0), pi_related], dim=0)

        weighted_pi = (pi_values * reliability.unsqueeze(1)).sum(dim=0, keepdim=True)

        return weighted_pi, reliability


class DefaultMixtureStrategy:
    def __init__(self, model, criterion, mixture_fn, reliability_fn, decay_fn):
        """This class implements a mixture strategy for combining target and related model
        scores. The strategy has three main components:

        1. A decay function that adjusts the influence of related models based on the number of training samples.
        2. A reliability function that computes the reliability scores of related models based on the training data.
        3. A mixture function that combines the target model scores with the related model scores using the reliability scores.
        """

        self.model = model
        self.criterion = criterion
        self.mixture_fn = mixture_fn
        self.reliability_fn = reliability_fn
        self.decay_fn = decay_fn

    def __call__(self, x_train, y_train, pi_target, pi_related, minimize):
        device = x_train.device

        reliability_scores = self.reliability_fn(
                x_train.unsqueeze(1), y_train.unsqueeze(1),
                minimize=minimize
            ).to(device)

        scores = self.mixture_fn(
            pi_target=pi_target,
            pi_related=pi_related,
            reliability_scores=reliability_scores,
            alpha=self.decay_fn(x_train.shape[0])
        )
        return scores, reliability_scores



def acq_mixture(target_scores, related_scores, reliability_scores, alpha=1):
    """
    Mixes the target and related scores based on their reliability scores.
    :param target_scores: Target scores (e.g., from a model).
    :param related_scores: Related scores (e.g., from other models).
    :param reliability_scores: Reliability scores for the related models.
    :param alpha: Mixing coefficient.
    :return: Mixed scores.
    """
    weights = 1 / (reliability_scores + 1e-8)  # Inverse with numerical stability
    weights /= weights.sum()  # Normalize to probability distribution

    # min-max normalize the scores; for related scores, we normalize across the first axis
    target_scores = (target_scores - target_scores.min()) / (target_scores.max() - target_scores.min())
    # related_scores = (related_scores - related_scores.min(axis=1, keepdims=True)[0]) / (
    #     related_scores.max(axis=1, keepdims=True)[0] - related_scores.min(axis=1, keepdims=True)[0]
    # )

    weighted_related = (related_scores * weights.reshape(1, -1, 1)).sum(axis=1, keepdims=True)
    mixed_scores = alpha * target_scores + (1 - alpha) * weighted_related

    return torch.clamp(mixed_scores.squeeze(), 0 + 1e-6, 1)

def weighted_average(target_logits, related_logits, reliability_scores, alpha=1):
    weights = 1 / (reliability_scores + 1e-8)  # Inverse with numerical stability
    weights /= weights.sum()  # Normalize to probability distribution

    weighted_related = (related_logits * weights.reshape(1, -1, 1)).sum(axis=1, keepdims=True)
    mixture_logits = alpha * target_logits + (1 - alpha) * weighted_related

    return mixture_logits

def argmin(target_logits, related_logits, reliability_scores, alpha=1):
    weights = torch.zeros_like(reliability_scores)
    weights[torch.argmin(reliability_scores)] = 1

    weighted_related = (related_logits * weights.reshape(1, -1, 1)).sum(axis=1, keepdims=True)
    mixture_logits = alpha * target_logits + (1 - alpha) * weighted_related
    return mixture_logits


def softmax_mixture(target_logits, related_logits, reliability_scores, temperature=1.0, alpha=1):
    softmax_weights = torch.softmax(-reliability_scores / temperature, dim=0)
    weighted_related = (related_logits * softmax_weights.reshape(1, -1, 1)).sum(axis=1,keepdims=True)

    mixture_logits = alpha * target_logits + (1 - alpha) * weighted_related
    return mixture_logits