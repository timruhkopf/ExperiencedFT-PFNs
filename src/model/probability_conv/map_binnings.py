import torch


import functools
import hashlib
import torch

def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

def cache_with_tensor_hash(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(E1, E2):
        key = (tensor_hash(E1), tensor_hash(E2))
        if key not in cache:
            cache[key] = func(E1, E2)
        return cache[key]
    return wrapper

@cache_with_tensor_hash
def build_coverage_matrix(E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
    """
    Compute fractional overlap of bins defined by two sets of edges.

    Args:
        E1: 1D tensor of sorted bin edges for the first binning (length n1+1)
        E2: 1D tensor of sorted bin edges for the second binning (length n2+1)

    Returns:
        (n1 x n2) tensor M where M[i, j] is the fraction of first-bin i
        covered by second-bin j.
    """
    if torch.any(E1[1:] <= E1[:-1]) or torch.any(E2[1:] <= E2[:-1]):
        raise ValueError("Edges must be strictly increasing.")

    # Bin starts/ends
    a1, b1 = E1[:-1], E1[1:]
    a2, b2 = E2[:-1], E2[1:]

    # Broadcast to compute pairwise overlaps
    left = torch.maximum(a1[:, None], a2[None, :])
    right = torch.minimum(b1[:, None], b2[None, :])
    overlap = torch.clamp(right - left, min=0.0)

    lengths1 = (b1 - a1)[:, None]  # shape: (n1, 1)
    frac_overlap = overlap / lengths1  # fractional overlap per old bin

    # Identify old bins completely left or right out of new bin edges:
    new_left_edge = a2[0]
    new_right_edge = b2[-1]

    # Boolean mask for old bins fully to the left or right of new binning:
    left_out_of_bounds = (b1 <= new_left_edge)  # old bin ends before new bins start
    right_out_of_bounds = (a1 >= new_right_edge)  # old bin starts after new bins end

    # For left out-of-bound old bins: zero all overlaps, assign full mass to first new bin
    frac_overlap[left_out_of_bounds, :] = 0
    frac_overlap[left_out_of_bounds, 0] = 1

    # For right out-of-bound old bins: zero all overlaps, assign full mass to last new bin
    frac_overlap[right_out_of_bounds, :] = 0
    frac_overlap[right_out_of_bounds, -1] = 1

    return frac_overlap


def make_kernel_grid(error_borders: torch.Tensor,
                     target_borders: torch.Tensor,
                     device: torch.device,
                     round_decimals: int = 3) -> torch.Tensor:
    """
    Construct kernel grid aligned with target_borders.

    Args:
        error_borders: 1D tensor of bin edges for error model
        target_borders: 1D tensor of target bin edges
        device: torch device
        round_decimals: rounding precision for left border

    Returns:
        kernel_grid: 1D tensor of aligned grid edges
    """
    left = error_borders[0]
    right = error_borders[-1]

    # now ensure we have a symmetric grid
    max_range =  torch.max(torch.abs(left), torch.abs(right) )
    left = -max_range
    right = max_range

    step = target_borders[1] - target_borders[0]
    rounded_left = torch.round(left, decimals=round_decimals).to(device)

    return torch.arange(rounded_left, right + step, step).to(device)


def project_probs_to_new_grid(error_logits: torch.Tensor,
                              error_borders: torch.Tensor,
                              kernel_grid: torch.Tensor,
                              return_logits: bool = False) -> torch.Tensor:
    """
    Project error model probabilities onto a new kernel grid.
    Args:
        error_logits: tensor of shape (..., n1) containing logits over error_borders bins
        error_borders: 1D tensor of bin edges for original error model
        kernel_grid: 1D tensor of bin edges for target grid
        return_logits: if True return log-space, else return probabilities
    Returns:
        error_probs_projected or error_logits_projected
    """
    overlap = build_coverage_matrix(error_borders, kernel_grid)  # (n1, n2)
    error_probs = torch.softmax(error_logits, dim=-1)  # (batch, n1)
    error_probs_projected = torch.matmul(error_probs, overlap)  # (batch, n2)

    # Normalize explicitly to be safe
    error_probs_projected = error_probs_projected / error_probs_projected.sum(dim=-1, keepdim=True)

    if return_logits:
        return torch.log(error_probs_projected.clamp(min=1e-12))
    else:
        return error_probs_projected


if __name__ == '__main__':

    import torch

    # Assuming error_logits, orig_bins, target_bins, projected_probs from previous example

    import matplotlib.pyplot as plt

    # Create original binning and target binning (for example purposes)
    device = torch.device("cpu")
    orig_bins = torch.linspace(-5, 5, steps=51, device=device)  # 50 bins edges in [-5,5]
    target_bins = torch.linspace(-4, 4, steps=33, device=device)  # 32 bins edges in [-4,4]
    # !!!! NOTICE, that in this example, the tail bins of the orig_bins fall out of the bounds of the target_bins,
    # !!!! The overlap matrix accounts for that by assigning the tail mass to its closest target
    # bin. This is proper behaviour from the perspective of PI!

    # Create gaussian distributions in original bins centers
    orig_bin_centers = 0.5 * (orig_bins[:-1] + orig_bins[1:])


    def gaussian_pdf(x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)


    # Parameters of Gaussians (means and std devs)
    params = [
        (0.0, 1.0),
        (-1.0, 0.5),
        (1.5, 1.2),
        (-3.0, 0.8)  # Adding a fourth Gaussian for variety
    ]

    # Create batch of logits by taking log of pdf (add tiny offset to avoid log(0))
    gaussians = []
    for mu, sigma in params:
        pdf_vals = gaussian_pdf(orig_bin_centers, mu, sigma)
        pdf_vals /= pdf_vals.sum()  # Normalize to sum=1 probability
        logits = torch.log(pdf_vals + 1e-12)
        gaussians.append(logits)

    error_logits = torch.stack(gaussians, dim=0)  # shape: (batch=3, n1=50)

    # Project from original binning to target binning
    projected_probs = project_probs_to_new_grid(
        error_logits, orig_bins, target_bins,
        return_logits=True
    )

    # Prepare to plot
    target_bin_centers = 0.5 * (target_bins[:-1] + target_bins[1:])

    orig_bin_centers = 0.5 * (orig_bins[:-1] + orig_bins[1:])
    orig_bin_widths = orig_bins[1:] - orig_bins[:-1]
    target_bin_centers = 0.5 * (target_bins[:-1] + target_bins[1:])
    target_bin_widths = target_bins[1:] - target_bins[:-1]

    plt.figure(figsize=(12, 6))

    colors = ['r', 'g', 'b', 'y']
    for i, color in enumerate(colors):
        # Compute original and projected probabilities
        orig_probs = torch.softmax(error_logits[i], dim=-1)
        proj_probs = torch.softmax(projected_probs[i], dim=-1)

        # Convert probability mass to density by dividing by bin widths
        orig_density = orig_probs / orig_bin_widths
        proj_density = proj_probs / target_bin_widths

        # Plot original as bars
        plt.bar(orig_bin_centers.cpu(), orig_density.cpu(), width=orig_bin_widths.cpu(),
                alpha=0.5, edgecolor=color, facecolor=color, label=f'Original Gaussian {i + 1}')

        # Plot projected as bars, slightly transparency and narrower for visibility
        plt.bar(target_bin_centers.cpu(), proj_density.cpu(), width=target_bin_widths.cpu(),
                alpha=0.3, edgecolor=color, facecolor='none', linewidth=2,
                label=f'Projected Gaussian {i + 1}')

    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Original and Projected Gaussian PDFs (Density)")
    plt.legend()
    plt.grid(True)
    plt.show()
