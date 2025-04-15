import torch


class DTrain(torch.utils.data.Dataset):

    def __init__(self, x, y, length=100, sequence_length_max=500):
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

    def __len__(self):
        return self.length

    def get_shuffled_data(self):
        shuffled_indices = torch.randperm(self.x.size(0))
        return self.x[shuffled_indices], self.y[shuffled_indices]

    def get_rnd_subset_initialization(self, size):
        """
        Get a random subset of the data for initialization.

        Args:
            size: The size of the subset to be returned.

        Returns:
            A random subset of the data.
        """
        x, y = self.get_shuffled_data()
        return x[:size], y[:size]

    def __getitem__(self, idx):
        # shuffle x and y in sequence dimension
        x_shuffled, y_shuffled = self.get_shuffled_data()

        x = x_shuffled
        y = y_shuffled

        return x.squeeze(1), y.squeeze(1)

def collate(batch, min_length=10, max_length=500):
    """
    Collate function to combine a batch of data into a single tensor.

    Args:
        batch: A list of tuples containing (x, y) data.

    Returns:
        A tuple of tensors (x, y).
    """

    size = torch.randint(min_length, max_length, (1,)).item()
    x = torch.stack([item[0][:size] for item in batch])
    y = torch.stack([item[1][:size] for item in batch])


    # (Sequence length, Batch size, hparameter dimension), (Sequence length, Batch size)
    return x.permute(1, 0, 2), y.permute(1, 0)

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