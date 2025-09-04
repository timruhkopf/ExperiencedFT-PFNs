import math

import torch
import torch.nn.functional as F

from ifbo import BarDistribution
from model.probability_conv.conv import batch_convolve_distributions
from model.probability_conv.map_binnings import make_kernel_grid, project_probs_to_new_grid


class DistributionConvolver:
    def __init__(self, chunk_batch_size=5):
        self.device = "cpu"
        self.chunk_batch_size = chunk_batch_size

    def _make_dirac(self, y_obs, borders, batch_shape):
        """
        Create a Dirac distribution at observed y_obs,
        represented as logits over the given borders.
        """
        idx = BarDistribution(borders=borders).map_to_bucket_idx(y_obs)
        T, B = batch_shape
        D = len(borders)

        logits = torch.full((T, B, D), -1e9, device=self.device)
        logits.scatter_(-1, idx.unsqueeze(-1), 0.0)  # δ(y - y_obs)
        return logits

    def convolve(self, A_logits, borders_A, B_logits, borders_B, reverse=False,
                 target_borders=None, padding=None):
        """
        Convolve two probability distributions (possibly on different grids).

        Parameters
        ----------
        A_logits:     (T, B, D1) logits of distribution A
        borders_A:    (D1,) tensor of bin borders for A
        B_logits:     (T, B, D2) logits of distribution B
        borders_B:    (D2,) tensor of bin borders for B
        reverse:      flip B if True (e.g. for inverse error model)
        """

        import torch
        import torch.nn.functional as F
        P_A = F.softmax(A_logits, dim=-1)
        P_B = F.softmax(B_logits, dim=-1)

        # homogenize binning if necessary
        if borders_A.shape != borders_B.shape or not torch.allclose(borders_A, borders_B):
            left = torch.min(borders_A[0], borders_B[0])
            right = torch.max(borders_A[-1], borders_B[-1])
            interval = torch.min(
                min(borders_A[1:] - borders_A[:-1]),
                min(borders_B[1:] - borders_B[:-1])
            ).item()
            kernel_grid = torch.arange(left, right, interval).to(self.device)

            new_B_logits = project_probs_to_new_grid(
                B_logits,
                borders_B,
                kernel_grid=kernel_grid,
                return_logits=True
            )
            P_B_old = P_B.clone()
            borders_B_old = borders_B.clone()
            P_B = torch.softmax(new_B_logits, dim=-1)
            borders_B = kernel_grid

            new_A_logits = project_probs_to_new_grid(
                A_logits,
                borders_A,
                kernel_grid=kernel_grid,
                return_logits=True
            )
            P_A_old = P_A.clone()
            borders_A_old = borders_A.clone()
            P_A = torch.softmax(new_A_logits, dim=-1)
            borders_A = kernel_grid

            # assert torch.allclose(P_A.sum(dim=-1) , P_A_old.sum(dim=-1))
            # assert torch.allclose(P_B.sum(dim=-1) , P_B_old.sum(dim=-1))

            if False:

                # Squeeze time dimension T, shapes become (B, kernel_dim)
                P_A_old = P_A_old.squeeze(0).cpu()
                P_A = P_A.squeeze(0).cpu()
                P_B_old = P_B_old.squeeze(0).cpu()
                P_B = P_B.squeeze(0).cpu()
                borders_A_old = borders_A_old.cpu()
                borders_B_old = borders_B_old.cpu()
                borders_A = borders_A.cpu()
                borders_B = borders_B.cpu()

                B = P_A.shape[0]  # batch size

                fig, axs = plt.subplots(B, 2, figsize=(14, 4 * B))

                if B == 1:
                    axs = axs.reshape(1, 2)

                for i in range(B):
                    # Plot old and new P_A (left column)
                    axs[i, 0].plot(borders_A_old[:-1], P_A_old[i], '--', label='Old P_A')
                    axs[i, 0].plot(borders_A[:-1], P_A[i], '-', label='New P_A')
                    axs[i, 0].set_title(f'Batch {i + 1} - Distribution A')
                    axs[i, 0].set_xlabel('Borders A')
                    axs[i, 0].set_ylabel('Probability')
                    axs[i, 0].legend()

                    # Plot old and new P_B (right column)
                    axs[i, 1].plot(borders_B_old[:-1], P_B_old[i], '--', label='Old P_B')
                    axs[i, 1].plot(borders_B[:-1], P_B[i], '-', label='New P_B')
                    axs[i, 1].set_title(f'Batch {i + 1} - Distribution B')
                    axs[i, 1].set_xlabel('Borders B')
                    axs[i, 1].set_ylabel('Probability')
                    axs[i, 1].legend()

                plt.tight_layout()
                plt.show()

        if reverse:
            # assert torch.allclose(borders_B, -borders_B.flip(dims=(-1,))), \
            #     "B borders must be symmetric around zero for reverse convolution"
            P_B = torch.flip(P_B, dims=(-1,))

        # P_A = P_A[:5]
        # P_B = P_B[:5]

        # flatten time/batch for convolution
        T_A, B_A, D_A = P_A.shape
        T_B, B_B, D_B = P_B.shape
        assert T_A == T_B and B_A == B_B, "Mismatch in batch/time dims"

        T= T_A
        B = B_A
        # Assuming T = number of time steps, B = batch size, D_A, D_B = feature dims
        p_a = P_A.view(-1, D_A)
        p_B = P_B.view(-1, D_B)

        if self.chunk_batch_size == -1:
            chunk_batch_size = p_a.shape[0]
        else:
            chunk_batch_size = self.chunk_batch_size

        chunks = []
        num_chunks = math.ceil(B / chunk_batch_size)
        for i in range(num_chunks):
            chunk_start = i*chunk_batch_size*T
            chunk_end = chunk_start + chunk_batch_size*T
            P_A_chunk = p_a[chunk_start:chunk_end]
            P_B_chunk = p_B[chunk_start:chunk_end]

            C_chunk, centers_conv, info = batch_convolve_distributions(
                A=P_A_chunk,
                B=P_B_chunk,
                edges_A=borders_A.cpu().numpy(),
                edges_B=borders_B.cpu().numpy(),
                device=self.device
            )
            chunks.append(C_chunk)
        C_batch = torch.cat(chunks, dim=0)

        # C_batch, centers_conv, info = batch_convolve_distributions(
        #     A=P_A.reshape(-1, D_A),
        #     B=P_B.reshape(-1, D_B),
        #     edges_A=borders_A.cpu().numpy(),
        #     edges_B=borders_B.cpu().numpy(),
        #     device=self.device
        # )
        C_batch = C_batch.view(T, B, -1)

                                                        # "matchull convolution")
        T, B = T_A, B_A  # batch shape
        convolved_logits = torch.log(C_batch.clamp(min=1e-12))
        centers_conv = torch.tensor(centers_conv, dtype=torch.float32).to(self.device)
        delta = (centers_conv[1:] - centers_conv[:-1]) / 2
        borders_conv = torch.cat([
            centers_conv[:1] - delta[0:1],
            centers_conv[:-1] + delta,
            centers_conv[-1:] + delta[-1:]
        ])

        if target_borders is not None:
            convolved_logits = project_probs_to_new_grid(
                convolved_logits,
                borders_conv,
                target_borders,
                return_logits=True,
                truncate=True
            )
            borders_conv = target_borders

        return convolved_logits, BarDistribution(borders=borders_conv)

    def convolve_dirac(
            self,
            y_obs, borders_y,
            error_logits, borders_error,
            batch_shape,
            reverse=True
    ):
        """
        Special case: convolve Dirac(y_obs) with error distribution.
        """
        dirac_logits = self._make_dirac(y_obs, borders_y, batch_shape)
        return self.convolve(
            A_logits=dirac_logits,
            borders_A=borders_y,
            B_logits=error_logits,
            borders_B=borders_error,
            reverse=reverse
        )

    def to(self, device):
        """
        Move the convolver to a specific device.
        """
        self.device = device
        return self


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch

    convolver = DistributionConvolver()

    borders_1 = torch.linspace(-3, 3, 900)
    borders_2 = torch.linspace(-5, 5, 1000)
    T, B = 1, 3


    def gauss_pdf(grid, mean, std):
        centers = (grid[:-1] + grid[1:]) / 2
        return torch.exp(-0.5 * ((centers - mean) / std) ** 2)


    means_1 = torch.tensor([0.0, 0.0, 0.3])
    stds_1 = torch.tensor([1., 0.5, 0.2])
    gauss_vals_1 = torch.stack(
        [gauss_pdf(borders_1, m.item(), s.item()) for m, s in zip(means_1, stds_1)], dim=0)
    gauss_vals_1 /= gauss_vals_1.sum(dim=-1, keepdim=True)
    logits_1 = torch.log(gauss_vals_1).unsqueeze(0)  # shape (T, B, D1)

    means_2 = torch.tensor([0., -1, 0.5])
    stds_2 = torch.tensor([1., 0.5, 0.6])
    gauss_vals_2 = torch.stack(
        [gauss_pdf(borders_2, m.item(), s.item()) for m, s in zip(means_2, stds_2)], dim=0)
    gauss_vals_2 /= gauss_vals_2.sum(dim=-1, keepdim=True)
    logits_2 = torch.log(gauss_vals_2).unsqueeze(0)  # shape (T, B, D2)

    # Numerical convolution
    convolved_logits, convolved_criterion = convolver.convolve(
        A_logits=logits_1, borders_A=borders_1,
        B_logits=logits_2, borders_B=borders_2,
        reverse=False
    )

    convolved_probs = torch.softmax(convolved_logits, dim=-1).squeeze(0)  # shape (B, D_conv)

    # Analytical convolution of Gaussians
    convolved_borders = convolved_criterion.borders
    convolved_centers = (convolved_borders[:-1] + convolved_borders[1:]) / 2  # shape (D_conv,)



    # For each batch compute analytical Gaussian
    analytical_gaussians = []
    for i in range(B):
        mean_conv = means_1[i] + means_2[i]
        std_conv = torch.sqrt(stds_1[i] ** 2 + stds_2[i] ** 2)
        gauss_vals = torch.exp(-0.5 * ((convolved_centers - mean_conv) / std_conv) ** 2)
        gauss_vals /= gauss_vals.sum()
        analytical_gaussians.append(gauss_vals)
    analytical_gaussians = torch.stack(analytical_gaussians, dim=0)  # shape (B, D_conv)

    # Plot
    fig, axes = plt.subplots(B, 1, figsize=(10, 4 * B))
    if B == 1:
        axes = [axes]

    x_1_centers = (borders_1[:-1] + borders_1[1:]) / 2
    x_2_centers = (borders_2[:-1] + borders_2[1:]) / 2

    for i in range(B):
        axes[i].plot(x_1_centers.cpu(), torch.exp(logits_1[0, i]).cpu(), label="Gaussian 1")
        axes[i].plot(x_2_centers.cpu(), torch.exp(logits_2[0, i]).cpu(), label="Gaussian 2")
        axes[i].plot(convolved_centers.cpu(), convolved_probs[i].cpu(),
                     label="Numerical Convolved")
        axes[i].plot(convolved_centers.cpu(), analytical_gaussians[i].cpu(), '--',
                     label="Analytical Convolved")
        axes[i].legend()
        axes[i].set_title(f"Batch {i + 1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Probability")

    plt.tight_layout()
    plt.show()
