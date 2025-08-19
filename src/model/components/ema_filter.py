import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def ema_conv_causal(x: torch.Tensor, alpha: float, truncate: int = None, bias_correction: bool = True) -> torch.Tensor:
    """
    Causal EMA via conv1d with correct alignment and optional bias correction.
    - Pads only on the LEFT (causal).
    - Reverses kernel because PyTorch conv1d is cross-correlation.
    - Optionally applies bias-correction so y0 == x0 and early-time shrinkage is removed.
    """
    T, N = x.shape
    if truncate is None:
        truncate = T

    # Build exponential kernel b_k = alpha * (1-alpha)^k, k=0..K-1 (causal weights for lags 0..K-1)
    coeffs = alpha * (1 - alpha) ** torch.arange(truncate, dtype=x.dtype, device=x.device)
    # Reverse for cross-correlation
    kernel = coeffs.flip(0).view(1, 1, -1)  # [out_channels=1, in_channels=1, K]

    # Input as (batch=N, channels=1, length=T)
    x_t = x.T.unsqueeze(1)

    # Causal padding: pad only on the LEFT by K-1
    x_pad = F.pad(x_t, (truncate - 1, 0))

    # Valid conv (no extra padding) with shared kernel across series
    y = F.conv1d(x_pad, kernel)  # shape (N,1,T)
    y = y[:, 0, :T].T  # (T,N)

    if bias_correction:
        # Divide by (1 - (1-alpha)^{t+1}) so startup bias disappears
        t = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(1)  # (T,1)
        correction = 1 - (1 - alpha) ** (t + 1)
        y = y / correction

    return y

if __name__ == '__main__':

    # --- Example data: 3 series ---
    T = 200
    torch.manual_seed(0)
    time = torch.arange(T, dtype=torch.float32)

    x1 = torch.sin(0.15 * time) + 0.4 * torch.randn(T)                # sine + noise
    x2 = 0.03 * time + 0.6 * torch.randn(T)                           # trend + noise
    x3 = torch.cumsum(0.2 * torch.randn(T), dim=0) + 0.01 * time      # noisy random walk + drift

    X = torch.stack([x1, x2, x3], dim=1)  # (T,3)

    alpha = 0.1
    Y = ema_conv_causal(X, alpha, truncate=100, bias_correction=True)

    # --- Plot: one chart per series (no subplots) ---
    titles = ["Sine + Noise", "Trend + Noise", "Random Walk + Drift"]
    for i in range(X.shape[1]):
        plt.figure(figsize=(10, 3.2))
        plt.plot(time.numpy(), X[:, i].numpy(), label="Original")
        plt.plot(time.numpy(), Y[:, i].numpy(), label=f"EMA (alpha={alpha})", linewidth=2)
        plt.title(titles[i])
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
