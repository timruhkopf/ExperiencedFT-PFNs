import torch

class Imputer:
    def __init__(self, model, criterion, imputation_mode):
        self.model = model
        self.criterion = criterion
        self.imputation_mode = imputation_mode

    def __call__(self, x_train, y_train, x_test: torch.Tensor, ) -> torch.Tensor:
        """
        Impute the y values for the training data from the target task under the prior context.
        """

        imputed_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=related_context.padding_mask
        )

        if self.imputation_mode == 'median':
            imputed_y = self.criterion.median(imputed_logits)
        elif self.imputation_mode == 'mean':
            imputed_y = self.criterion.mean(imputed_logits)
        elif self.imputation_mode == 'sample':
            raise NotImplementedError("Here probably is a shaper error!")
            imputed_y = []
            for b in range(imputed_logits.shape[1]):
                imputed_y.append(sample_logits(imputed_logits.squeeze(0), n_samples=1,
                                      borders=self.criterion.borders))
            imputed_y = imputed_y.squeeze(-1)

            # probs = imputed_logits.softmax(-1)
            # bins = self.criterion.borders
            # bucket_middle = (bins[:-1] + bins[:-1] + self.criterion.bucket_widths) / 2
            # sampled_indices = torch.stack([
            #     torch.multinomial(probs[:, i, :], 1) for i in range(probs.shape[1])
            # ], dim=1)
            #
            # # Remove the last dimension for direct indexing
            # # Shape: (T, n_related_tasks)
            # sampled_indices = sampled_indices.squeeze(-1)
            #
            # # Gather the corresponding bin values
            # # Shape: (T, n_related_tasks, num_bars) if bins is 2D, else (T, n_related_tasks)
            # imputed_y = bucket_middle[sampled_indices]
        else:
            raise ValueError(f"Unknown imputation mode: {self.imputation_mode}")

        return imputed_y


def sample_logits(logits: torch.Tensor, n_samples: int, borders) -> torch.Tensor:
    """
    Sample from the logits distribution.

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

    return sample_y.T  # Add batch dimension