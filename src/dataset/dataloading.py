import torch


class DTrain(torch.utils.data.Dataset):

    def __init__(self, x, y, padding_mask=None, length=100, sequence_length_max=500):
        """

        Just a shallow dataset class, that is supposed to shuffle the datapoints in the sequence dimension

        :param x:
        :param y:
        :param length:
        :param sequence_length_max:
        """
        self.x = x
        self.y = y
        self.length = length
        self.sequence_length_max = sequence_length_max

        if padding_mask is None:
            self.padding_mask = torch.ones(x.size(1), x.size(0), dtype=torch.bool).to(x.device)

        else:
            self.padding_mask = padding_mask

        self.valid_mask = ~self.padding_mask
        self.num_valid = self.valid_mask.sum(dim=1)
        self.n_tasks = self.padding_mask.shape[0]

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        # sample at most to the fixed size of the smallest dataset. (collate will subset this)
        size=min(self.num_valid).item()

        # sample a fixed number of valid tokens from the sequence
        valid_mask = self.valid_mask
        batch, seq_len = valid_mask.shape
        device = valid_mask.device
        sampled_indices = torch.full((batch, size), -1, dtype=torch.long, device=device)
        for i in range(batch):
            valid_indices = torch.nonzero(valid_mask[i], as_tuple=False).flatten()
            num_valid = valid_indices.size(0)
            if num_valid >= size:
                chosen = torch.randperm(num_valid, device=device)[:size]
                sampled_indices[i] = valid_indices[chosen]
            elif num_valid > 0:
                # If not enough, sample with replacement
                chosen = torch.randint(0, num_valid, (size,), device=device)
                sampled_indices[i] = valid_indices[chosen]
            # else: leave as -1 (no valid tokens)

        # collect the sampled indices
        batch, n_samples = sampled_indices.shape
        seq_length, batch2, dim = self.x.shape
        assert batch == batch2

        # Prepare indices for gather:
        # We want to index x[ sampled_indices[b, i], b, : ] for each b, i
        # So we need to build a batch index of shape [batch, n_samples]
        batch_idx = torch.arange(batch, device=self.x.device).unsqueeze(1).expand(-1,n_samples)  # [batch, n_samples]

        # Now, sampled_indices and batch_idx can be used to index x
        # But PyTorch advanced indexing wants all indices as 1D, so flatten:
        gathered_x = self.x[sampled_indices, batch_idx, :]  # [batch, n_samples, dim]
        gathered_y = self.y[sampled_indices, batch_idx]  # [batch, n_samples]
        return gathered_x, gathered_y

def collate(batch, min_length=10, max_length=500):
    """
    Collate function to combine a batch of data into a single tensor.

    Args:
        batch: A list of tuples containing (x, y) data.

    Returns:
        A tuple of tensors (x, y).
    """

    size = torch.randint(min_length, max_length, (1,)).item()

    # fixme: we get a 4 dim tensor: (dataloader.batch, n_tasks, n_samples, dim)
    x = torch.stack([item[0][:size] for item in batch])
    y = torch.stack([item[1][:size] for item in batch])

    n_tasks, T, dim = x.shape[1:]
    # (Sequence length, Batch size, hparameter dimension), (Sequence length, Batch size)
    # return x.permute(1, 0, 2), y.permute(1, 0)
    x = x.reshape(-1, T, dim)
    y = y.reshape(-1, T)


    return x.permute(1,0,2), y.permute(1,0)

if __name__ == '__main__':
    from functools import partial
    # Example usage
    x = torch.randn(1000, 10)  # Example data
    y = torch.randn(1000, 1)   # Example labels

    dataset = DTrain(x, y, length=100)
    collate_fn = partial(collate, min_length=10, max_length=500)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    for batch_x, batch_y in dataloader:
        print("Batch X shape:", batch_x.shape , "Batch Y shape:", batch_y.shape)