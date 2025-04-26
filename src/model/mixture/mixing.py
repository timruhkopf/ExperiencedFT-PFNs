import torch


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
    weighted_related = (related_logits * softmax_weights.reshape(1, -1, 1)).sum(axis=1,
                                                                                keepdims=True)

    mixture_logits = alpha * target_logits + (1 - alpha) * weighted_related
    return mixture_logits