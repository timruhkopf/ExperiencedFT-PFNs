import ifbo
import torch


# Try to fix the batch issue in model.forward for nll evaluation:
class ModifiedFTPFN(ifbo.surrogate.FTPFN):
    def batch_forward(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x_train (torch.Tensor): context points, shape (batch_size x n_context x features)
            y_train (torch.Tensor): context values, shape (batch_size x n_context)
            x_test (torch.Tensor): query points, shape (batch_size x n_query x features)

        Returns:
            torch.Tensor: logits for query points, shape (batch_size x n_query)
        """
        self._check_input(x_train, y_train, x_test)
        batch_size = x_train.shape[0]

        # Handle ID modifications for batched inputs
        if x_train.shape[1] == 0:
            x_test[..., 0] = 0
        else:
            # Add 1 to IDs if min ID is 0 (batch-wise operation)
            zero_mask = x_train[..., 0].min(dim=1).values == 0
            x_train[zero_mask, :, 0] += 1
            x_test[zero_mask, :, 0] += 1

            # Create batch-specific masks for ID validation
            for i in range(batch_size):
                valid_ids = x_train[i, :, 0]
                x_test[i, :, 0] = torch.where(
                    torch.isin(x_test[i, :, 0], valid_ids),
                    x_test[i, :, 0],
                    torch.zeros_like(x_test[i, :, 0])
                )

        single_eval_pos = x_train.shape[1]
        query_batch_size = 2000  # Per-batch elements to process
        n_batches = (x_test.shape[1] + query_batch_size - 1) // query_batch_size

        results = []
        for i in range(n_batches):
            start = i * query_batch_size
            end = min((i + 1) * query_batch_size, x_test.shape[1])

            # Concatenate context and query points along sequence dimension
            x_batch = torch.cat([
                x_train,  # (batch_size, n_context, features)
                x_test[:, start:end, :]  # (batch_size, n_queries, features)
            ], dim=1).unsqueeze(2)  # Add dummy dimension if needed by your model

            y_batch = y_train.unsqueeze(2)  # (batch_size, n_context, 1)

            # Process through model (adjust based on actual model expectations)
            result = self.model(
                (x_batch, y_batch),
                single_eval_pos=single_eval_pos
            )
            results.append(result)

        return torch.cat(results, dim=1)

if __name__ == '__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedFTPFN(version="0.0.1", device=device)

    single_eval_pos=700
    batch = ifbo.priors.ftpfn_prior.get_batch(
        batch_size=2,
        seq_len=1000,  # maximum number of observations per task for training. Default: 1000.
        num_features=12,
        # sample the dimension of HPs with Uniform(1, num_features-1). Default: 12.
        single_eval_pos=single_eval_pos
    )

    # normalize x
    x = batch.x
    x = (x-x.min())/(x.max()-x.min())

    x_train = x[:single_eval_pos].permute(1,0,2)
    y_train = batch.y.permute(1,0)
    x_test = x[single_eval_pos:].permute(1,0,2)


    logits = model.batch_forward(x_train=x_train, y_train=y_train, x_test=x_test)