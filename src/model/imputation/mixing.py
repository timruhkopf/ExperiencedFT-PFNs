import torch


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
    related_scores = (related_scores - related_scores.min(axis=1, keepdims=True)[0]) / (
        related_scores.max(axis=1, keepdims=True)[0] - related_scores.min(axis=1, keepdims=True)[0]
    )

    weighted_related = (related_scores * weights.reshape(1, -1, 1)).sum(axis=1, keepdims=True)
    mixed_scores = alpha * target_scores + (1 - alpha) * weighted_related

    return torch.clamp(mixed_scores.squeeze(), 0 + 1e-6, 1)