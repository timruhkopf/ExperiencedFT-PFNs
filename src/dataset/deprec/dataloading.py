from typing import List

import torch


class DTrain(torch.utils.data.Dataset):

    def __init__(self, x, y, padding_mask=None, length=100):
        """

        Just a shallow dataset class, that is supposed to shuffle the datapoints in the sequence dimension

        :param x:
        :param y:
        :param length:

        """
        self.x = x
        self.y = y
        self.length = length

        if padding_mask is None:
            self.padding_mask = torch.zeros(x.size(1), x.size(0), dtype=torch.bool).to(x.device)

        else:
            self.padding_mask = padding_mask

        B, T = padding_mask.shape

        samplable = []
        for b in range(B):
            samplable.append(torch.where(torch.logical_not(padding_mask[b, :]))[0])

        self.samplable_indices = samplable

        self.valid_mask = ~self.padding_mask
        self.num_valid = self.valid_mask.sum(dim=1)
        self.n_tasks = self.padding_mask.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        T, B, D = self.x.shape

        # Store the sampled indices for each batch element
        x = torch.zeros((T, B, D), device=self.x.device)
        y = torch.zeros((T, B), device=self.x.device)

        for b, size in enumerate(self.num_valid):
            # shuffle only the valid tokens (not the padded ones)
            sampled_indices = torch.randperm(size)
            x[:size, b] = self.x[sampled_indices, b, :]
            y[:size, b] = self.y[sampled_indices, b]

        return x, y, self.padding_mask


def collate(batch, min_length=10):
    """
    Collate function to combine a batch of data into a single tensor.

    Args:
        batch: A list of tuples containing (x, y) data.

    Returns:
        A tuple of tensors (x, y).
    """
    x0 = batch[0][0]
    T, n_tasks, D = x0.shape

    padding_per_task = batch[0][2]
    valid_mask = ~padding_per_task
    count_per_task = valid_mask.sum(dim=1)

    # we cheat here, by saying that some points are padded (which they are not,
    # but we don't want them to be seen in this example)

    padding_per_task = batch[0][2]
    valid_mask = ~padding_per_task
    count_per_task = valid_mask.sum(dim=1)

    # Random sizes for each (batch, task) entry
    sizes = torch.randint(min_length, count_per_task[0], (len(batch), n_tasks))

    # Create a range for time steps
    time = torch.arange(T).expand(len(batch), n_tasks, T)

    # Broadcast and compare to mark as observed (False = observed, True = padded)
    padding = time >= sizes.unsqueeze(-1)

    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    # padding = torch.stack([item[2] for item in batch])
    # permute to get x: (batch, n_samples, n_tasks, dim), y: (batch, n_samples, n_tasks)
    # return x.permute(0, 2, 1, 3), y.permute(0, 2, 1), padding
    return x, y, padding


if __name__ == '__main__':
    from functools import partial

    # Example usage
    x = torch.randn(1000, 10)  # Example data
    y = torch.randn(1000, 1)  # Example labels

    dataset = DTrain(x, y, length=100)
    collate_fn = partial(collate, min_length=10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    for batch_x, batch_y in dataloader:
        print("Batch X shape:", batch_x.shape, "Batch Y shape:", batch_y.shape)
