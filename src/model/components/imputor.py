import torch
import torch
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import qmc


class PriorImputer:
    def __init__(self, imputation_mode):
        self.imputation_mode = imputation_mode
        self.imputation_strategies = {
            'median': self._impute_median,
            'mean': self._impute_mean,
            'joint_surrogate_sample': self._impute_joint_surrogate_sample,
            'margin_sample': self._impute_margin_sample
        }

    def __post_init__(self, parent_model, model, criterion, related_context, logger, device):
        self.parent_model = parent_model
        self.model = model
        self.criterion = criterion
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]
        self.logger = logger
        self.device = device

    def collect_support_set(self, x_train, x_test, y_train, inc, **kwargs):
        T, B, hpD = x_train.shape
        # 2. collect a random support set

        n_support_samples = kwargs.get('n_support_samples', None)
        support = kwargs.get('support', 'sobol')

        if n_support_samples is None:
            n_support_samples = 10 * hpD  # heuristic

        if support == 'sobol':
            sobol_engine = qmc.Sobol(d=hpD, scramble=True)
            sobol_sample = sobol_engine.random(n=n_support_samples)

        elif support == 'lhd':
            lhs_engine = qmc.LatinHypercube(d=hpD)
            sobol_sample = lhs_engine.random(n=n_support_samples)

        support_x = torch.tensor(sobol_sample, device=self.device).float().unsqueeze(1)
        support_x[..., 0] = 999.  # hp id dimension in IFBO!

        if x_train.shape[1] != support_x.shape[1]:
            support_x = support_x.repeat(1, x_train.shape[1], 1)

        return support_x

    def __call__(self, x_train, y_train, x_test, **kwargs) -> torch.Tensor:
        if self.imputation_mode not in self.imputation_strategies:
            raise ValueError(f"Unknown imputation mode: {self.imputation_mode}")
        return self.imputation_strategies[self.imputation_mode](x_train, y_train, x_test, **kwargs)

    def _impute_median(self, x_train, y_train, x_test, **kwargs):
        imputed_logits = self.model(
            (torch.cat([x_train, x_test], dim=0), y_train),
            single_eval_pos=x_train.shape[0]
        )
        return self.criterion.median(imputed_logits)

    def _impute_mean(self, x_train, y_train, x_test, **kwargs):
        imputed_logits = self.model(
            (torch.cat([x_train, x_test], dim=0), y_train),
            single_eval_pos=x_train.shape[0])
        return self.criterion.mean(imputed_logits)

    def _impute_margin_sample(self, x_train, y_train, x_test, **kwargs):
        imputed_logits = self.model(
            (torch.cat([x_train, x_test], dim=0), y_train),
            single_eval_pos=x_train.shape[0]
        )
        n_samples = kwargs.get("n_samples", 1)
        imputed_y = torch.stack([
            sample_logits(
                imputed_logits[:, b, :].squeeze(0),
                n_samples=n_samples,
                borders=self.criterion.borders
            )
            for b in range(imputed_logits.shape[1])
        ], dim=1)
        return imputed_y

    def _impute_joint_surrogate_sample(self, x_train, y_train, x_test, **kwargs) -> torch.Tensor:
        """
        Impute the y values for the training data from the target task under the prior context.
        """

        if x_test is None:
            x_test = self.collect_support_set(
                x_train, x_test, y_train,
                inc=None,
                n_support_samples=kwargs.get('n_support_samples', 100),
            )

        support_x = x_test

        logits = self.model(
            (torch.cat([x_train, x_test], dim=0), y_train),
            single_eval_pos=x_train.shape[0],
        )

        n_samples = kwargs.get("n_samples", 1)

        # sampling for each task independently but n_samples at a time
        mu_post_list = []
        cov_post_list = []
        std_post_list = []
        full_sample_list = []
        for b in range(x_train.shape[1]):
            mu_post, cov_post = gp_condition(
                # trim the id dimension!
                x_train[:, b, 1:],  # (T, N_train_feats-1)
                y_train[:, b],  # (T,)
                x_test[:, b, 1:],  # (T, N_test_feats-1)
                lengthscale=0.5,
                variance=1.0,
                noise_var=1e-4
            )
            std_post = torch.sqrt(torch.diag(cov_post) + 1e-8)

            mvn = torch.distributions.MultivariateNormal(
                mu_post,
                covariance_matrix=cov_post + 1e-8 * torch.eye(len(x_test[:, b, :]))
            )

            full_sample = mvn.sample((n_samples,))

            mu_post_list.append(mu_post)
            cov_post_list.append(cov_post)
            std_post_list.append(std_post)
            full_sample_list.append(full_sample)

        mu_post = torch.stack(mu_post_list, dim=1).unsqueeze(-1)  # (T, B, ...)
        std_post = torch.stack(std_post_list, dim=1).unsqueeze(-1)  # (T, B)
        full_sample = torch.stack(full_sample_list, dim=1).permute(2, 1, 0)  # (T, B, n_samples)

        # Compute quantiles of each joint sampled value under marginal of the MVN at that point
        quantiles = torch.distributions.Normal(
            loc=mu_post,
            scale=std_post
        ).cdf(full_sample)  # (T, B, n_samples)



        # collect the y values corresponding to the quantile of the joint sample
        imputed_y = icdf(logits, quantiles, self.model.criterion.borders)

        # # Plot MVN posterior, joint sample, and quantiles of the joint sample
        # plt.figure(figsize=(12, 6))
        # plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Training data')
        # plt.plot(x_test.numpy(), mu_post.numpy(), 'b-', label='Posterior mean')
        # plt.fill_between(x_test.flatten().numpy(), (mu_post - 2 * std_post).numpy(),
        #                  (mu_post + 2 * std_post).numpy(), color='blue', alpha=0.2,
        #                  label='95% CI')
        # plt.plot(x_test.numpy(), full_sample.numpy(), 'g--', lw=2,
        #          label='Joint posterior sample')
        # sc = plt.scatter(x_test.numpy(), mu_post.numpy(), c=quantiles.numpy(),
        #                  cmap='coolwarm',
        #                  label='Quantile of joint sample under marginal')
        # plt.colorbar(sc, label='Marginal Quantile')
        # plt.legend()
        # plt.xlabel('X')
        # plt.ylabel('Function value')
        # plt.title('GP Posterior with joint sample and marginal quantiles')
        # plt.show()

        # self.plot_sample(x_train, y_train, imputed_y, support_x)

        return support_x, imputed_y

    def plot_sample(self, x_train, y_train, imputed_y, support_x):
        import plotly.graph_objs as go
        from plotly.offline import plot

        # assume x_train, y_train, and imputed_y are torch tensors from your snippet
        # also assuming you trimmed the ID dim (so shapes are: x_train: (T, D=2), y_train: (T,), imputed_y: (T,))
        # if still batched: slice [ :, 0, :] for simplicity

        # Example for plotting single batch (B=1)
        x_tr = x_train[:, 0, 1:].cpu().numpy() if x_train.ndim == 3 else x_train[
            :, 1:].cpu().numpy()
        y_tr = y_train[:, 0].cpu().numpy() if y_train.ndim == 2 else y_train.cpu().numpy()

        # Use the first imputed sample (n_samples=1)
        SAMPLE_IDX = 3
        y_imp = imputed_y[:, 0, SAMPLE_IDX].cpu().numpy() if imputed_y.ndim == 3 else imputed_y.cpu(

        ).numpy()

        # Build Plotly figure
        fig = go.Figure()

        # Training data points
        fig.add_trace(go.Scatter3d(
            x=x_tr[:, 0],  # D=3 -> two features + y, using first two input dims
            y=x_tr[:, 1],
            z=y_tr,
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.7),
            name='Training Data'
        ))

        # Imputed sample curve / points
        fig.add_trace(go.Scatter3d(
            x=support_x[:, 0, 1].cpu().numpy(),
            y=support_x[:, 0, 2].cpu().numpy(),
            z=y_imp,
            mode='markers',
            marker=dict(size=4, color='orange'),
            line=dict(color='orange', width=2),
            name='Sampled Imputation'
        ))

        # Layout
        fig.update_layout(
            title='Training Data and Sampled Imputation',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Output'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Display offline
        plot(fig)


def sample_logits(logits: torch.Tensor, n_samples: int, borders) -> torch.Tensor:
    """
    Sample from the marginal logits distributions.
    May introduce ruggedness due to independent sampling.

    Args:
        logits (torch.Tensor): Logits tensor of shape (T, n_related_tasks, num_bars).
        n_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: Sampled values of shape (n_samples, T, n_related_tasks).
    """
    probs = logits.softmax(-1)
    sampled_indices = torch.multinomial(probs, n_samples, replacement=True)

    # Remove the last dimension for direct indexing
    sampled_indices = sampled_indices.squeeze(-1)

    # Calculate the middle of each bin
    bucket_middle = (borders[:-1] + borders[1:]) / 2
    # Gather the corresponding bin values
    sample_y = bucket_middle[sampled_indices]

    return sample_y  # Add batch dimension


def icdf(logits: torch.Tensor, left_prob: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    T, B, D = logits.shape
    _, _, S = left_prob.shape

    probs = logits.softmax(-1)  # (T, B, D)
    cumprobs = torch.cumsum(probs, dim=-1)  # (T, B, D)

    # Flatten T and B so cumprobs has shape (T*B, D)
    cumprobs_flat = cumprobs.reshape(-1, D)  # (T*B, D)
    left_prob_flat = left_prob.reshape(-1, S)  # (T*B, S)

    # For each (T*B), searchsorted over cumprobs for left_prob values (T*B, S)
    idx = torch.searchsorted(cumprobs_flat, left_prob_flat, right=False)  # (T*B, S)
    idx = idx.clamp(0, D - 1)

    # Pad cumprobs_flat with zero on the left to get left border cumulative prob
    cumprobs_padded = torch.cat(
        [torch.zeros(cumprobs_flat.size(0), 1, device=logits.device), cumprobs_flat],
        dim=-1,
    )  # (T*B, D+1)

    # Gather cumulative probabilities just before the indexed bin
    cum_left = torch.gather(cumprobs_padded, 1, idx + 1)  # (T*B, S)
    rest_prob = left_prob_flat - cum_left  # (T*B, S)

    # Get borders at idx and idx+1, expand borders to allow indexing for (T*B, S)
    borders_exp = borders.unsqueeze(0)  # (1, D+1)
    left_border = borders[idx]  # advanced indexing, shape (T*B, S)
    right_border = borders[idx + 1]

    # Gather probs at indices
    prob_at_idx = torch.gather(probs.reshape(-1, D), 1, idx)  # (T*B, S)

    quantile = left_border + (right_border - left_border) * rest_prob / prob_at_idx

    # Reshape back to (T, B, S)
    quantile = quantile.reshape(T, B, S)

    return quantile


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (N1, N2, D)
    dist_sq = torch.sum(diff ** 2, dim=2)
    return variance * torch.exp(-0.5 * dist_sq / lengthscale ** 2)


def make_psd(matrix, jitter=1e-4):
    matrix = 0.5 * (matrix + matrix.T)
    matrix += jitter * torch.eye(matrix.size(0))
    return matrix


def gp_condition(X_train, y_train, X_test, lengthscale=1.0, variance=1.0,
                 noise_var=1e-4):
    K = rbf_kernel(X_train, X_train, lengthscale, variance)
    K_s = rbf_kernel(X_test, X_train, lengthscale, variance)
    K_ss = rbf_kernel(X_test, X_test, lengthscale, variance)

    K_jittered = make_psd(K + noise_var * torch.eye(len(X_train)))
    L = torch.linalg.cholesky(K_jittered)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L)

    mu_post = K_s @ alpha
    v = torch.linalg.solve_triangular(L, K_s.T, upper=False)
    cov_post = K_ss - v.T @ v
    cov_post = make_psd(cov_post)

    return mu_post.squeeze(-1), cov_post
