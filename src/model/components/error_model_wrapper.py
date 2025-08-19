import math
from copy import deepcopy

import pfns4bo
import torch
import torch.nn.functional as F

from pfns4bo.bar_distribution import BarDistribution

import logging

from model.components.conv import batch_convolve_distributions
from model.components.map_binnings import make_kernel_grid, project_probs_to_new_grid

logger = logging.getLogger(__name__)


class WrappedErrorModel:
    def __init__(self, target_criterion, device, error_model=None):
        self.error_model = error_model if error_model is not None else torch.load(pfns4bo.bnn_model,
                                                                                  weights_only=False)
        self.error_model.eval()
        self.device = device
        self.error_model.to(device)

        self.error_borders = deepcopy(self.error_model.criterion.borders)
        self.target_borders = target_criterion.borders
        self.target_criterion = target_criterion

    def __getattr__(self, name):
        return getattr(self.error_model, name)

    def __setattr__(self, name, value):
        if name == 'error_model':
            super().__setattr__(name, value)
        else:
            setattr(self.error_model, name, value)

    def __call__(self, x_train, y_error, x_test, padding=None, std_fwd=False):
        # Here we exaggerate the difference to meet the high resolution range [-2.5, 2.5]
        # of the model -- we will need to undo this later to communicate the
        # result distribution in the target binning for convolution.
        # now we determine the factor by whcih we need to scale y_error, such that
        # we do not exceed the error model's resolution range


        # 1) Fit the y->z scaling for THIS batch (or cache from train to avoid leakage)
        if std_fwd:
            original_range = torch.min(y_error, dim=0), torch.max(y_error, dim=0)
            M = y_error.abs().max().clamp_min(1e-6)  # raw-space symmetric radius
            self.alpha = 2.5 / M  # y -> z scale

            # 2) Push targets & borders into model space
            z_error = self.alpha * y_error  # z in [-2.5, 2.5]
            self.error_model.criterion.borders = self.error_borders / self.alpha  # push-forward once
            self.error_model.criterion.bucket_widths = self.error_model.criterion.borders[1:] - \
                                                       self.error_model.criterion.borders[:-1]

            y_error = z_error

        # 3) Forward
        error_logits = self.error_model(
            (torch.cat([x_train, x_test], dim=0), y_error),
            single_eval_pos=x_train.shape[0],
        )

        if self.error_model.criterion.borders.shape !=  self.target_borders.shape \
                or not torch.all(self.error_model.criterion.borders == self.target_borders):


            kernel_grid = make_kernel_grid(
                error_borders=self.error_borders,
                target_borders=self.target_borders,
                device=self.device,
                round_decimals=3
            )

            # TODO: since we always have the same border mapping, we could cache the overlap matrix
            error_logits = project_probs_to_new_grid(
                error_logits,
                self.error_borders,
                kernel_grid=kernel_grid,
                return_logits=True
            )

            self.error_model.criterion.borders = kernel_grid
            self.error_model.criterion.bucket_widths = kernel_grid[1:] - kernel_grid[:-1]

        else:
            kernel_grid = self.target_borders

        error_probs = torch.softmax(error_logits, dim=-1)

        return error_probs, kernel_grid, error_logits

    def dirac_forward(self, dirac_y, dirac_x, x_train, y_error, reverse=True):
        """
        The main idea here is, that we have observed y --> so it is a
        dirac probability mass. Now we could shift the error to it directly.
        Since the convolution is FFT and thus efficient, we can avoid errorprone
        implementations by simply expanding y into binary vectors and convolve
        """
        idx = BarDistribution(borders=self.target_borders).map_to_bucket_idx(dirac_y)
        T, B = dirac_y.shape[:2]
        D = len(self.target_borders)

        # Expand y into one-hot logits ("Dirac probabilities")
        #    logits = big negative values everywhere except at idx position
        logits = torch.full((T, B, D), -1e9, device=dirac_y.device)
        logits.scatter_(-1, idx.unsqueeze(-1), 0.0)  # 0 at the Dirac positions (softmax = 1.0)

        # Convolve "Dirac probabilities" with error kernel
        #   Use your convolution helper (should accept batch shape)
        #   convolve_probs_with_error expects logits and kernel
        #   You may need to pass proper bin borders
        logits_borders = self.target_borders

        convolved_logits, error_logits, bar_dist = self.convolve_probs_with_error(
            logits, logits_borders,
            x_train, y_error, dirac_x,
            padding=None,
            reverse=reverse  # already handled above
        )

        return convolved_logits, error_logits, bar_dist

    def convolve_probs_with_error(
            self,
            logits, logits_borders,
            x_train, y_error, x_test,
            padding=None,
            reverse=False
    ):
        """
        Here we want to convolve the probability distribution with the error model.
        This will basically shift and scale the logits of the target model
        :param probs:
        :param probs_bins:
        :param kernel:
        :param kernel_bins:
        :return:
        """
        # check for x_train, x_test [:,:,0] that these values are [0, 1]
        assert torch.all(x_train[:, :, 0] >= 0) and torch.all(x_train[:, :, 0] <= 1), \
            "x_train[:, :, 0] should be in [0, 1]"
        assert torch.all(x_test[:, :, 0] >= 0) and torch.all(x_test[:, :, 0] <= 1), \
            "x_test[:, :, 0] should be in [0, 1]"

        error_probs_kernel, kernel_grid, error_logits = self.__call__(
            x_train=x_train, y_error=y_error, x_test=x_test, padding=padding
        )

        if reverse:
            torch.flip(error_probs_kernel, dims=(-1,))

        probs = F.softmax(logits, dim=-1)
        T, B, D = probs.shape
        T, B, D2 = error_probs_kernel.shape

        # flatten the time and batch dimensions for convolution
        C_batch, centers_conv, info = batch_convolve_distributions(
            A=probs.view(-1, D),
            B=error_probs_kernel.view(-1, D2),
            edges_A=logits_borders.cpu().numpy(),
            edges_B=kernel_grid.cpu().numpy(),
            device=self.device
        )

        # plot_convs_facet(
        #     probs.view(-1, D)[:4].detach().numpy(),
        #     error_probs_kernel.view(-1, D2)[:4].detach().numpy(),
        #     C_batch[:4].detach().numpy(),
        #     edges_A=logits_borders.cpu().numpy(),
        #     edges_B=kernel_grid.cpu().numpy(),
        #     centers_conv= centers_conv,
        #     info=info,
        #     xlim=(-1, 1),
        # )

        convolved_logits = torch.log(C_batch.clamp(min=1e-12)).reshape(T, B, centers_conv.shape[0])  # convert back to logits
        centers_conv = torch.tensor(centers_conv, dtype=torch.float32).to(self.device)


        # we need to fix the borders of the convolved distribution, by
        # taking the centers_conv and taking the borders from it
        borders = torch.zeros(centers_conv.shape[0] + 1, dtype=torch.float32).to(self.device)
        borders[:-1] = centers_conv - (self.target_borders[1] - self.target_borders[0]) / 2
        borders[-1] = centers_conv[-1] + (self.target_borders[1] - self.target_borders[0]) / 2


        return convolved_logits, error_logits, BarDistribution(borders=borders)




def convolve_probs_with_error(probs, probs_bins, kernel, kernel_bins, plot=False, **kwargs):
    batch_size, length = probs.shape
    kernel_size = kernel.shape[1]

    # Original input shape: (batch_size, 1, length)
    p_orig_t = probs.view(batch_size, 1, length)

    # Flip kernels for convolution
    p_shift_flipped = torch.flip(kernel, dims=[1]).view(kernel.shape[0], 1, kernel_size)

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


def plot_buckets(borders):
    bandwidths = borders[1:] - borders[:-1]

    import numpy as np
    import matplotlib.pyplot as plt

    probabilities = bandwidths.numpy()
    probabilities /= probabilities.sum()  # normalize if not already

    # CDF calculation: cumulative sum
    cdf = np.concatenate([[0], np.cumsum(probabilities)])

    plt.figure(figsize=(8, 4))
    plt.step(borders, cdf, where='post',
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


if __name__ == '__main__':
    from pfns4bo import bnn_model

    target_criterion = BarDistribution(borders=torch.linspace(0, 1, 1001)).to('cpu')
    error_model = torch.load(pfns4bo.bnn_model, weights_only=False)
    error_model.eval()
    n = 20  # observed samples in error

    x_train = torch.rand(100, 1, 1)  # 10 samples, 1 batch, 2 features
    y_prior = x_train[:, :, 0] * 0.5 + torch.randn(100, 1) * 0.02  #
    # y_prior is a linear function of x_train
    y_target = x_train[:, :, 0] * 0.3 + 0.1  # y_target is also a linear function of x_train
    y_error = y_target[:n, :] - y_prior[:n, :]

    x_test = torch.linspace(0, 1, 1000).to('cpu').reshape(1000, 1, 1)  # test points

    # UNPROJECTED ERROR MODEL:  -----------------------------
    error_logits = error_model(
        (
            torch.cat([x_train[:n], x_test], dim=0),
            y_error
        ),
        single_eval_pos=x_train[:n].shape[0],
    )

    lower = error_model.criterion.icdf(error_logits, 0.05).detach().numpy().flatten()
    upper = error_model.criterion.icdf(error_logits, 0.95).detach().numpy().flatten()
    mean = error_model.criterion.mean(error_logits).detach().numpy().flatten()


    we_model = torch.load(pfns4bo.bnn_model, weights_only=False)
    we_model.eval()
    werror_model = WrappedErrorModel(
        target_criterion=target_criterion,
        device='cpu',
        error_model=we_model
    )

    error_probs, kernel_grid, error_logits = werror_model(
        x_train[:n], y_error, x_test, padding=None
    )
    # error_logits = torch.log(error_probs_kernel.clamp(min=1e-12))

    w_lower = werror_model.criterion.icdf(error_logits, 0.05)
    w_upper = werror_model.criterion.icdf(error_logits, 0.95)
    w_mean = werror_model.criterion.mean(error_logits)

    # CONVOLUTION WITH ERROR MODEL: ------------------------------
    prior_model = torch.load(pfns4bo.bnn_model, weights_only=False)
    prior_logits = prior_model(
        (torch.cat([x_train, x_test], dim=0), y_prior),
        single_eval_pos=x_train.shape[0],
    )
    prior_lower = prior_model.criterion.icdf(prior_logits, 0.05)
    prior_upper = prior_model.criterion.icdf(prior_logits, 0.95)
    prior_mean = prior_model.criterion.mean(prior_logits)

    # WRAPPED PRIOR MODEL ------
    wp_model = torch.load(pfns4bo.bnn_model, weights_only=False)
    wp_model.eval()

    wprior_model=WrappedErrorModel(
        target_criterion=target_criterion,
        device='cpu',
        error_model=wp_model
    )

    _, _, wprior_logits = wprior_model(
        x_train=x_train, y_error=y_prior, x_test=x_test)


    w_prior_lower = wprior_model.criterion.icdf(wprior_logits, 0.05)
    w_prior_upper = wprior_model.criterion.icdf(wprior_logits, 0.95)
    w_prior_mean = wprior_model.criterion.mean(wprior_logits)

    convolved_logits, error_logits, convolved_criterion = werror_model.convolve_probs_with_error(
        logits=wprior_logits, logits_borders=wprior_model.criterion.borders,
        x_train=x_train[:n], y_error=y_error, x_test=x_test,
    )

    convolved_lower = convolved_criterion.icdf(convolved_logits, 0.05)
    convolved_upper = convolved_criterion.icdf(convolved_logits, 0.95)
    convolved_mean = convolved_criterion.mean(convolved_logits)



    # PLOT THE RESULTS ------------------------------
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_test_np = x_test.squeeze(-1).detach().cpu().numpy().flatten()
    x_train_np = x_train.squeeze(-1).detach().cpu().numpy().flatten()

    y_prior_np = y_prior.detach().cpu().numpy().flatten()
    y_target_np = y_target.detach().cpu().numpy().flatten()

    plt.figure(figsize=(8, 6))

    # Target and Prior
    plt.plot(x_train_np, y_target_np, 'o', label="Target", color="tab:blue")
    plt.plot(x_train_np, y_prior_np, 'o', label="Prior", color="tab:orange")
    # Plot the observed error samples
    plt.plot(x_train_np[:n], y_error.detach().cpu().numpy().flatten(),
             'o', label="Observed Error", color="tab:red")
    plt.plot(x_train_np[:n], y_prior[:n].detach().cpu().numpy().flatten(),
             'x', label="Observed Error", color="tab:red")

    # ORIGINAL ERROR MODEL BOUNDS
    plt.fill_between(x_test_np, lower, upper, alpha=0.2, color="tab:green",
                     label="95% CI")
    plt.plot(x_test_np, mean, label="Mean", color="tab:green", linewidth=2)

    # PROJECTED ERROR MODEL BOUNDS
    # plt.fill_between(x_test_np, new_bound_lower, new_bound_upper, alpha=0.2, color="tab:purple",
    #                  label="Projected 95% CI")
    # plt.plot(x_test_np, new_bound_mean, label="Projected Mean", color="tab:purple", linewidth=2)

    # WRAPPED ERROR MODEL BOUNDS
    plt.fill_between(x_test_np, w_lower.detach().cpu().numpy().flatten(),
                        w_upper.detach().cpu().numpy().flatten(), alpha=0.2, color="tab:cyan",
                        label="Wrapped 95% CI")
    plt.plot(x_test_np, w_mean.detach().cpu().numpy().flatten(), label="Wrapped Mean",
             color="tab:cyan", linewidth=2)

    # PRIOR MODEL BOUNDS
    plt.fill_between(x_test_np, prior_lower.detach().cpu().numpy().flatten(),
                     prior_upper.detach().cpu().numpy().flatten(), alpha=0.2, color="tab:purple",
                     label="Prior 95% CI")
    plt.plot(x_test_np, prior_mean.detach().cpu().numpy().flatten(), label="Prior Mean",
             color="tab:purple", linewidth=2)

    # sns.rugplot(y=prior_model.criterion.borders.cpu().numpy(),)
    # sns.rugplot(y=target_criterion.borders.cpu().numpy(), color="tab:red")

    # WRAPPED PRIOR MODEL BOUNDS
    plt.fill_between(x_test_np, w_prior_lower.detach().cpu().numpy().flatten(),
                        w_prior_upper.detach().cpu().numpy().flatten(), alpha=0.2, color
                        ="tab:orange",
                        label="Wrapped Prior 95% CI")
    plt.plot(x_test_np, w_prior_mean.detach().cpu().numpy().flatten(), label="Wrapped Prior Mean",
                color="tab:orange", linewidth=2)

    # CONVOLVED BOUNDS
    plt.fill_between(x_test_np, convolved_lower.detach().cpu().numpy().flatten(),
                     convolved_upper.detach().cpu().numpy().flatten(), alpha=0.2, color="tab:gray",
                     label="Convolved 95% CI")
    plt.plot(x_test_np, convolved_mean.detach().cpu().numpy().flatten(), label="Convolved Mean",
                color="tab:gray", linewidth=2)

    # plot the hline (0)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Target, Prior, Prediction Intervals, and Mean")
    plt.grid(True)
    plt.show()

    # # plot the prior_model's border widths distribution
    # plot_buckets(prior_model.criterion.borders)
    # plot_buckets( target_criterion.borders)

    print()
