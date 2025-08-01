from copy import deepcopy

import pfns4bo
import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pfns4bo.bar_distribution import BarDistribution


class WrappedErrorModel:
    def __init__(self, target_borders, device, error_model=None):
        self.error_model = error_model if error_model is not None else torch.load(pfns4bo.bnn_model,
                                                                                  weights_only=False)
        self.error_model.eval()
        self.device = device
        self.error_model.to(device)

        self.criterion: BarDistribution = self.error_model.criterion
        self.error_borders = deepcopy(self.error_model.criterion.borders)
        self.target_borders = target_borders

    def __getattr__(self, name):
        return getattr(self.error_model, name)

    def __setattr__(self, name, value):
        if name == 'error_model':
            super().__setattr__(name, value)
        else:
            setattr(self.error_model, name, value)

    def __call__(self, x_train, y_error, x_test, padding=None):
        # Here we exaggerate the difference to meet the high resolution range [-2.5, 2.5]
        # of the model -- we will need to undo this later to communicate the
        # result distribution in the target binning for convolution.
        # now we determine the factor by whcih we need to scale y_error, such that
        # we do not exceed the error model's resolution range
        FACTOR = 2.5 / max(y_error.abs().max(), 1e-6)  # avoid division by zero
        y_error = FACTOR * y_error

        # forward of the model
        error_logits = self.error_model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_error
            ),
            single_eval_pos=x_train.shape[0],
            # fixme: this model is not capable of accepting padding masks yet!
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(self.num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        self.error_model.criterion.borders = self.error_borders / FACTOR

        # scaling here affects the size of the kernel and the cost of the conv.
        left = min(self.error_model.criterion.icdf(error_logits[:, 0, :], 0.01))
        right = max(self.error_model.criterion.icdf(error_logits[:, 0, :], 0.99))

        kernel_grid = torch.arange(
            left, right,
            step=(self.target_borders[1] - self.target_borders[0]).item()
        ).to(self.device)

        # given the prior logits and error logits are differently binned distributions
        # we need to interpret the error bins and adjust probabiltiy mass of the prior logits
        # This projection can be done by computing the fractional overlap of each original bin
        # with each common bin and distributing the original bin’s probability accordingly.
        # now let us move the error logits into the target borders
        error_probs_kernel = project_probs_to_common_bins_batch(
            F.softmax(error_logits, dim=-1),
            self.error_model.criterion.borders,
            kernel_grid
        )

        return error_probs_kernel, kernel_grid, error_logits

    def dirac_forward(self, x_train, y_error, dirac_x, dirac_y, padding=None, reverse=False):
        """
        Forward pass for a Dirac delta distribution.
        """
        # Here we assume that dirac_x and dirac_y are already in the correct format
        # for the error model.

        dirac_y_error_logits, kernel_grid, _ = self.__call__(x_train, y_error, dirac_x, padding)

        if reverse:
            raise NotImplementedError("Reverse convolution is not implemented yet, but"
                                      "it is simple: just reverse the dirac kernel and convolve "
                                      "again")
            # TODO also flip the error_logits kernel (for debugging purposes)

        # we need to determine into which bin the dirac_y valls, and shift the
        #  entire dirac logits by this index
        idx = self.criterion.map_to_bucket_idx(dirac_y)
        shifted_y = torch.zeros_like(dirac_y_error_logits)

        # TODO find out in which direction the individual errors (positive / negative)
        #  and shift the logits accordingly

        raise NotImplementedError("Dirac forward is not implemented yet.")

    def convolve_probs_with_error(self, logits, x_train, y_error, x_test, padding=None,
                                  reverse=False):
        """
        Here we want to convolve the probability distribution with the error model.
        This will basically shift and scale the logits of the target model
        :param probs:
        :param probs_bins:
        :param kernel:
        :param kernel_bins:
        :return:
        """

        error_probs_kernel, kernel_grid, error_logits = self.__call__(
            x_train, y_error, x_test, padding
        )

        if reverse:
            raise NotImplementedError("Reverse convolution is not implemented yet, but"
                                      "it is simple: just reverse the kernel and convolve again")

        probs = F.softmax(logits, dim=-1)
        T, B, D = probs.shape

        # flatten the time and batch dimensions for convolution
        convolved_logits = convolve_probs_with_error(
            # probs, probs_bins, kernel, kernel_bins
            probs=probs.view(-1, D),
            probs_bins=self.target_borders,
            kernel=error_probs_kernel.view(-1, error_probs_kernel.shape[-1]),
            kernel_bins=kernel_grid

        ).reshape(T, B, -1)

        return convolved_logits, error_logits


def project_probs_to_common_bins_batch(orig_probs, orig_bounds, target_bounds):
    """
    Project batched probability distributions defined on orig_bounds to target_bounds
    by fractional overlap of bins in PyTorch.

    Args:
        orig_probs   : Tensor of shape (..., N) -- probabilities over original bins.
        orig_bounds  : 1D tensor of length N+1 -- bin edges of original distribution.
        target_bounds: 1D tensor of length M+1 -- bin edges of target distribution.

    Returns:
        projected_probs : Tensor of shape (..., M) -- probabilities projected onto target bins.
    """
    # orig_probs: (..., N)
    orig_shape = orig_probs.shape
    N = orig_shape[-1]
    M = target_bounds.shape[0] - 1

    # Expand bins for vectorized overlap calculation:
    # orig lefts and rights shape: (N, 1)
    orig_lefts = orig_bounds[:-1].unsqueeze(1)  # (N,1)
    orig_rights = orig_bounds[1:].unsqueeze(1)  # (N,1)
    # target lefts and rights shape: (1, M)
    target_lefts = target_bounds[:-1].unsqueeze(0)  # (1, M)
    target_rights = target_bounds[1:].unsqueeze(0)  # (1, M)

    # Calculate overlaps (N x M)
    overlaps = torch.clamp(
        torch.min(orig_rights, target_rights) - torch.max(orig_lefts, target_lefts),
        min=0.0)  # (N,M)
    orig_widths = (orig_rights - orig_lefts)  # (N,1)
    fractions = overlaps / orig_widths  # (N,M)

    # Move orig_probs last dim (N) to front to do batch matmul:
    # orig_probs reshaped to (-1, N)
    orig_probs_flat = orig_probs.reshape(-1, N)  # (B, N)

    # Multiply: (B, N) @ (N, M) => (B, M)
    projected_flat = torch.matmul(orig_probs_flat, fractions)  # (B, M)

    # Normalize so projected probabilities sum to 1 (for each batch)
    projected_flat /= projected_flat.sum(dim=1, keepdim=True)

    # Reshape back to original batch dims + M
    projected_shape = orig_shape[:-1] + (M,)
    projected_probs = projected_flat.reshape(projected_shape)

    debug = False
    if debug:
        plt.hist(projected_probs[0, 0].numpy(), bins=target_bounds, alpha=0.5,
                 label='Projected Probs')
        plt.show()

    return projected_probs


def convolve_probs_with_error(probs, probs_bins, kernel, kernel_bins, plot=False, **kwargs):
    batch_size, length = probs.shape
    kernel_size = kernel.shape[1]

    # Original input shape: (batch_size, 1, length)
    p_orig_t = probs.view(batch_size, 1, length)

    # Flip kernels for convolution
    p_shift_flipped = torch.flip(kernel, dims=[1]).view(batch_size, 1, kernel_size)

    # Now, merge batch into channels dimension by transposing:
    # Input: (batch_size, 1, length) -> (1, batch_size, length)
    p_orig_t_merged = p_orig_t.permute(1, 0, 2)  # (1, batch_size, length)

    # Weight already has shape (batch_size, 1, kernel_size)
    # To match input's channels, reshape kernels as (batch_size, 1, kernel_size)
    # Perform conv1d with groups = batch_size
    p_convolved = F.conv1d(
        p_orig_t_merged,  # input channels == batch_size
        p_shift_flipped,
        # weight shape must be (out_channels, in_channels/groups, kernel_size); here out_channels=batch_size, in_channels/groups=1
        padding=kernel_size - 1,
        groups=batch_size
    )

    # p_convolved shape: (1, batch_size, output_length)
    # reshape back to (batch_size, output_length)
    p_convolved = p_convolved.permute(1, 0, 2).view(batch_size, -1)
    p_convolved /= p_convolved.sum(dim=1, keepdim=True)  # probability norm

    # calculate the new bins of the convolved distribution
    min_kernel = kernel_bins[0]
    max_kernel = kernel_bins[-1]
    step = probs_bins[1] - probs_bins[0]
    prob_min = probs_bins[0]
    prob_max = probs_bins[-1]
    L = probs.shape[1]
    K = kernel.shape[1]
    conv_len = L + K - 1

    # Construct bin edges for original, kernel, and convolved (centered bins)
    # bins_probs = np.arange(prob_min, prob_max + step, step)  # Should have length L
    # bins_kernel = np.arange(min_kernel, max_kernel + step, step)  # length K
    bins_convolved = torch.arange(
        prob_min + min_kernel,
        prob_max + max_kernel + step,
        step
    )  # length conv_len

    # Calculate start and end indices for cropping (center-crop)
    start = (kernel_size - 1) // 2
    end = start + L

    # Crop convolved output & adjust the probability mass by adding the missing mass
    # to the left and right edges
    p_convolved_cropped = p_convolved[:, start:end]
    missing_prob_right = p_convolved[:, end:].sum(dim=1)
    missing_prob_left = p_convolved_cropped[:, :start].sum(dim=1)
    p_convolved_cropped[:, start] += missing_prob_left
    p_convolved_cropped[:, -1] += missing_prob_right

    if plot:
        plot_convolution(probs, probs_bins, kernel, kernel_bins, bins_convolved, start, end,
                         **kwargs)

    logits_convolved = torch.log(p_convolved_cropped.clamp(min=1e-12))

    return logits_convolved


def plot_convolution(probs, probs_bins, kernel, kernel_bins, p_convolved,
                     p_convolved_cropped, bins_convolved, start, end, idx=0):
    # Convert tensors to numpy for plotting
    probs_0 = probs[idx].numpy()
    kernel_0 = kernel[idx].numpy()
    p_convolved_0 = p_convolved[idx].numpy()
    p_convolved_cropped_0 = p_convolved_cropped[idx].numpy()

    # bin centers for plotting
    centers_probs_bins = (probs_bins[:-1] + probs_bins[1:]) / 2
    centers_kernel_bins = (kernel_bins[:-1] + kernel_bins[1:]) / 2
    centers_bins_convolved = (bins_convolved[:-1] + bins_convolved[1:]) / 2

    plt.figure(figsize=(12, 7))
    plt.plot(centers_probs_bins, probs_0, label='Original probs')
    plt.plot(centers_kernel_bins, kernel_0, label='Error probs (kernel)')
    plt.plot(centers_bins_convolved[:-2].numpy(), p_convolved_0, label='Full Convolved')

    cropped_bins = centers_bins_convolved[start:end]
    plt.plot(cropped_bins, p_convolved_cropped_0, label='Cropped Convolved')

    plt.title('Comparison of Original, Kernel, and Convolved Distributions')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.xlim([bins_convolved[0], bins_convolved[-1]])
    plt.show()


def plot_buckets(error_model, error_borders):
    bandwidths = error_model.criterion.bucket_widths

    import numpy as np
    import matplotlib.pyplot as plt

    probabilities = bandwidths.numpy()
    probabilities /= probabilities.sum()  # normalize if not already

    # CDF calculation: cumulative sum
    cdf = np.concatenate([[0], np.cumsum(probabilities)])

    plt.figure(figsize=(8, 4))
    plt.step(error_borders, cdf, where='post',
             label="CDF")
    plt.xlabel("Value")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_rescaled_data(y_target, y_error, error_borders):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert tensors to numpy arrays
    y_target_np = y_target.numpy().flatten()
    y_error_np = y_error.numpy().flatten()
    error_borders_np = error_borders.numpy().flatten()

    plt.figure(figsize=(8, 6))

    # Plot overlapping histograms
    plt.hist(y_target_np, bins=30, alpha=0.5, label='y_target', color='blue',
             edgecolor='black')
    plt.hist(y_error_np, bins=30, alpha=0.5, label='y_error', color='red',
             edgecolor='black')

    # Add rug plot for error_borders
    sns.rugplot(error_borders_np, color='green', height=0.05)

    plt.title('Overlapping Histograms with Rug plot of error_borders')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()


def plot_sth(y_target, y_hat, error_borders):
    # Convert tensors to numpy arrays
    y_target_np = y_target.numpy().flatten()
    y_hat_np = y_hat.numpy().flatten() / 16  # for median we can just divide!
    error_borders_np = error_borders.numpy().flatten()

    plt.figure(figsize=(8, 6))

    # Plot overlapping histograms
    plt.hist(y_target_np, bins=30, alpha=0.5, label='y_target', color='blue',
             edgecolor='black')
    plt.hist(y_hat_np, bins=30, alpha=0.5, label='y_hat_median', color='red',
             edgecolor='black')

    # Add rug plot for error_borders
    sns.rugplot(error_borders_np, color='green', height=0.01)

    plt.title('Overlapping Histograms with Rug plot of error_borders')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()


def plot_example_binning_transform():
    import numpy as np
    import matplotlib.pyplot as plt

    # Original probability distribution and bin edges
    orig_probs = np.array([0.05, 0.15, 0.3, 0.2, 0.1, 0.2])
    orig_bounds = np.array([0, 1, 2, 3, 4, 5, 6])  # 6 bins

    # Target bin edges (non-uniform widths)
    target_bounds = np.array([0, 0.5, 2.5, 3, 4.5, 6])  # 5 bins

    N = len(orig_probs)
    M = len(target_bounds) - 1

    # Step 1: Compute overlaps between each original and target bin
    orig_lefts = orig_bounds[:-1][:, None]  # (N, 1)
    orig_rights = orig_bounds[1:][:, None]  # (N, 1)
    target_lefts = target_bounds[:-1][None, :]  # (1, M)
    target_rights = target_bounds[1:][None, :]  # (1, M)

    # Overlap lengths for each (orig_bin, target_bin) pair
    overlaps = np.clip(
        np.minimum(orig_rights, target_rights) - np.maximum(orig_lefts, target_lefts),
        0, None
    )  # shape (N, M)

    orig_widths = orig_rights - orig_lefts  # (N, 1)
    fractions = overlaps / orig_widths  # (N, M)

    # Step 2: Redistribute probabilities using the fractions matrix
    # (orig_probs shape (N,), fractions (N, M))
    projected_probs = orig_probs @ fractions  # shape (M,)

    # Step 3: Normalize (optional—should already sum to 1, but for safety)
    projected_probs /= projected_probs.sum()

    # --- Visualization ---
    bin_centers_orig = (orig_bounds[:-1] + orig_bounds[1:]) / 2
    bin_centers_proj = (target_bounds[:-1] + target_bounds[1:]) / 2

    plt.figure(figsize=(8, 5))

    # Plot original histogram
    plt.bar(bin_centers_orig, orig_probs, width=1, alpha=0.7, label='Original', color='royalblue',
            edgecolor='black')

    # Plot projected histogram (shifted a bit for clarity)
    widths_proj = target_bounds[1:] - target_bounds[:-1]
    plt.bar(target_bounds[:-1], projected_probs,
            width=widths_proj,
            align='edge',
            alpha=0.6,
            label='Projected',
            color='orange',
            edgecolor='black')

    # Draw original and target bin edges
    for b in orig_bounds:
        plt.axvline(b, color='blue', ls='--', lw=1, alpha=0.25)
    for b in target_bounds:
        plt.axvline(b, color='orange', ls=':', lw=1, alpha=0.5)

    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Redistribution of Histogram Probabilities to New Bins')
    plt.tight_layout()
    plt.show()
