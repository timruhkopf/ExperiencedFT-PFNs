import numpy as np
import torch
import matplotlib.pyplot as plt
from math import ceil, log2


# -------------------------
# Utilities
# -------------------------
def next_pow2(x: int) -> int:
    return 1 << (int(ceil(log2(x))) if x > 0 else 0)


def interp_batch(y: torch.Tensor, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
    """
    Vectorized linear interpolation: y shape (batch, n_old),
    x_old shape (n_old,), x_new shape (n_new,).
    Extrapolates with 0 outside x_old range.
    """
    # x_old and x_new are 1D tensors on same device as y
    device = y.device
    x_old = x_old.to(device)
    x_new = x_new.to(device)

    # bucketize returns indices in [0, n_old], subtract 1 to get left index
    idx = torch.bucketize(x_new, x_old) - 1
    idx = idx.clamp(0, len(x_old) - 2)  # ensure idx+1 valid

    x0 = x_old[idx]  # (n_new,)
    x1 = x_old[idx + 1]  # (n_new,)
    y0 = y[:, idx]  # (batch, n_new)
    y1 = y[:, idx + 1]  # (batch, n_new)

    denom = (x1 - x0)
    # avoid division by zero if x1==x0 (shouldn't for proper bins)
    denom = torch.where(denom == 0, torch.tensor(1.0, device=device), denom)
    w = ((x_new - x0) / denom).unsqueeze(0)  # (1, n_new)
    return y0 + w * (y1 - y0)


# -------------------------
# Main convolution function
# -------------------------
def batch_convolve_distributions(A: torch.Tensor,
                                 B: torch.Tensor,
                                 edges_A: np.ndarray,
                                 edges_B: np.ndarray,
                                 device: str = 'cpu'):
    """
    Batch convolution of PDFs A and B (A: (batch, nA), B: (batch, nB)).
    edges_A: (nA+1,), edges_B: (nB+1,)
    Returns:
        C: (batch, valid_len) convolved PDFs (densities)
        centers_conv: numpy array of length valid_len (x positions for C)
        info: dict with Δ and other helpers
    Notes:
      - Uses Δ = min(Δ_A, Δ_B) as uniform spacing for resampling.
      - Output discrete convolution length = nA_u + nB_u - 1 (valid_len).
      - conv sample k corresponds to center = centerA0 + centerB0 + k*Δ.
      - final C is clamped at 0 and normalized so integral ≈ 1.
    """
    device = torch.device(device)
    A = A.to(device)
    B = B.to(device)

    edges_A = np.asarray(edges_A, dtype=float)
    edges_B = np.asarray(edges_B, dtype=float)

    centers_A = 0.5 * (edges_A[:-1] + edges_A[1:])
    centers_B = 0.5 * (edges_B[:-1] + edges_B[1:])

    centers_A_t = torch.tensor(centers_A, dtype=torch.float32, device=device)
    centers_B_t = torch.tensor(centers_B, dtype=torch.float32, device=device)

    # bin widths (assume roughly uniform per-histogram)
    ΔA = float(np.median(np.diff(edges_A)))
    ΔB = float(np.median(np.diff(edges_B)))
    Δ = min(ΔA, ΔB)

    # Build uniform grids (centers) for A and B with spacing Δ
    a_min, a_max = float(edges_A[0]), float(edges_A[-1])
    b_min, b_max = float(edges_B[0]), float(edges_B[-1])

    nA_u = int(np.round((a_max - a_min) / Δ))
    nB_u = int(np.round((b_max - b_min) / Δ))
    # ensure at least 1
    nA_u = max(1, nA_u)
    nB_u = max(1, nB_u)

    # Uniform centers: start at edge + Δ/2
    centers_A_u = (a_min + (np.arange(nA_u) + 0.5) * Δ).astype(np.float32)
    centers_B_u = (b_min + (np.arange(nB_u) + 0.5) * Δ).astype(np.float32)

    centers_A_u_t = torch.tensor(centers_A_u, dtype=torch.float32, device=device)
    centers_B_u_t = torch.tensor(centers_B_u, dtype=torch.float32, device=device)

    # Interpolate PDFs onto uniform grids
    A_u = interp_batch(A, centers_A_t, centers_A_u_t)  # (batch, nA_u)
    B_u = interp_batch(B, centers_B_t, centers_B_u_t)  # (batch, nB_u)

    # Ensure they are densities (integrate ~1)
    # (numerical safety) renormalize by sum * Δ
    A_u = A_u / (A_u.sum(dim=1, keepdim=True) * Δ + 1e-12)
    B_u = B_u / (B_u.sum(dim=1, keepdim=True) * Δ + 1e-12)

    # Convolution length and pad for FFT (use next pow2 for speed)
    valid_len = nA_u + nB_u - 1
    pad_len = next_pow2(valid_len)

    # FFT (real) with padding to pad_len
    A_padded = torch.nn.functional.pad(A_u, (0, pad_len - nA_u))
    B_padded = torch.nn.functional.pad(B_u, (0, pad_len - nB_u))
    A_fft = torch.fft.rfft(A_padded)
    B_fft = torch.fft.rfft(B_padded)
    C_fft = A_fft * B_fft
    C_full = torch.fft.irfft(C_fft, n=pad_len)  # (batch, pad_len)

    C_valid = C_full[:, :valid_len] * Δ  # scale by Δ for continuous convolution

    # Clamp small negatives, renormalize
    C_valid = C_valid.clamp(min=0.0)
    C_valid = C_valid / (C_valid.sum(dim=1, keepdim=True) * Δ + 1e-12)

    # Output centers: centerA0 + centerB0 + k*Δ
    centerA0 = centers_A_u[0]  # a_min + Δ/2
    centerB0 = centers_B_u[0]  # b_min + Δ/2
    centers_conv = (centerA0 + centerB0) + Δ * np.arange(valid_len)  # numpy array

    info = {'Δ': Δ,
            'nA_u': nA_u, 'nB_u': nB_u, 'valid_len': valid_len,
            'centers_A_u': centers_A_u, 'centers_B_u': centers_B_u,
            'centerA0': centerA0, 'centerB0': centerB0}

    return C_valid, centers_conv, info


def plot_convs_facet(A, B, C, edges_A, edges_B, centers_conv, info, max_cols=3, xlim=None):
    centers_A = 0.5 * (edges_A[:-1] + edges_A[1:])
    centers_B = 0.5 * (edges_B[:-1] + edges_B[1:])
    batch_size = A.shape[0]
    ncols = min(max_cols, int(np.ceil(np.sqrt(batch_size))))
    nrows = int(np.ceil(batch_size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    Δ = info['Δ']

    for i in range(batch_size):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.plot(centers_A, A[i], label='A (orig)', lw=1.5)
        ax.plot(centers_B, B[i], label='B (orig)', lw=1.5)

        ax.plot(centers_conv, C[i], label='Convolved (FFT)', lw=2)

        ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(min(centers_A[0], centers_B[0], centers_conv[0]) - 1,
                    max(centers_A[-1], centers_B[-1], centers_conv[-1]) + 1)



    # turn off unused axes
    for j in range(batch_size, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # -------------------------
    # Example: generate batch Gaussians and test
    # -------------------------
    def gaussian_batch_params(batch_size, edges, mu_range=(-5, 5), sigma_range=(0.5, 2.5),
                              seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        centers = 0.5 * (edges[:-1] + edges[1:])
        centers_t = torch.tensor(centers, dtype=torch.float32)
        mu = torch.empty(batch_size).uniform_(*mu_range)
        sigma = torch.empty(batch_size).uniform_(*sigma_range)
        pdf = torch.exp(
            -0.5 * ((centers_t.unsqueeze(0) - mu.unsqueeze(1)) / sigma.unsqueeze(1)) ** 2)
        pdf = pdf / (pdf.sum(dim=1, keepdim=True) * (edges[1] - edges[0]))
        return pdf, mu.numpy(), sigma.numpy(), centers  # return centers for plotting


    # Create example data
    batch_size = 5
    nA = 100
    nB = 150
    edges_A = np.linspace(-10, 10, nA + 1)
    edges_B = np.linspace(-8, 8, nB + 1)

    A, muA, sigmaA, centersA = gaussian_batch_params(batch_size, edges_A, mu_range=(-6, 2),
                                                     sigma_range=(0.6, 1.5), seed=1)
    B, muB, sigmaB, centersB = gaussian_batch_params(batch_size, edges_B, mu_range=(-2, 6),
                                                     sigma_range=(0.4, 1.2), seed=2)

    # Convert to torch
    A_t = A
    B_t = B

    # Convolve
    C_batch, centers_conv, info = batch_convolve_distributions(A_t, B_t, edges_A, edges_B,
                                                               device='cpu')


    # -------------------------
    # Plotting: facet grid overlaying A, B, C and analytic Gaussian sum
    # -------------------------
    def plot_facet(A, B, C, edges_A, edges_B, centers_conv, muA, sigmaA, muB, sigmaB, max_cols=3):
        centers_A = 0.5 * (edges_A[:-1] + edges_A[1:])
        centers_B = 0.5 * (edges_B[:-1] + edges_B[1:])
        batch_size = A.shape[0]
        ncols = min(max_cols, int(np.ceil(np.sqrt(batch_size))))
        nrows = int(np.ceil(batch_size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes)

        Δ = info['Δ']

        for i in range(batch_size):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.plot(centers_A, A[i].numpy(), label='A (orig)', lw=1.5)
            ax.plot(centers_B, B[i].numpy(), label='B (orig)', lw=1.5)

            ax.plot(centers_conv, C[i].numpy(), label='Convolved (FFT)', lw=2)

            # analytic Gaussian sum (for comparison)
            mu_sum = muA[i] + muB[i]
            sigma_sum = np.sqrt(sigmaA[i] ** 2 + sigmaB[i] ** 2)
            fine_x = np.linspace(centers_conv[0] - 3 * Δ, centers_conv[-1] + 3 * Δ, 800)
            analytic = np.exp(-0.5 * ((fine_x - mu_sum) / sigma_sum) ** 2)
            analytic /= (analytic.sum() * (fine_x[1] - fine_x[0]))  # normalize continuous approx
            ax.plot(fine_x, analytic, '--', label='Analytic Gaussian sum', alpha=0.8)

            ax.set_title(
                f'Batch {i}\nmuA={muA[i]:.2f},σA={sigmaA[i]:.2f} | muB={muB[i]:.2f},σB={sigmaB[i]:.2f}')
            ax.legend()
            ax.set_xlim(min(centers_A[0], centers_B[0], centers_conv[0]) - 1,
                        max(centers_A[-1], centers_B[-1], centers_conv[-1]) + 1)

        # turn off unused axes
        for j in range(batch_size, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis('off')

        plt.tight_layout()
        plt.show()


    plot_facet(A, B, C_batch, edges_A, edges_B, centers_conv, muA, sigmaA, muB, sigmaB)
