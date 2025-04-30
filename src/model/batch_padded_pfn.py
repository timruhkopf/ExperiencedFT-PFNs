from pathlib import Path

import torch

from ifbo.priors.prior import Batch


class MyBatch:
    """
    A batch of data with padding for the train set.
    """

    def __init__(self, x, y, single_eval_pos, query_x=None, query_y=None,
                 padding_mask=None, **kwargs):
        self.x = x
        self.y = y
        self.query_x = query_x
        self.query_y = query_y
        self.single_eval_pos = single_eval_pos
        self.padding_mask = padding_mask

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return (f"MyBatch(x={self.x.shape}, y={self.y.shape}, query_x="
                f"{self.query_x.shape if isinstance(self.query_x, torch.Tensor) else self.query_x}," \
                f"query_y="
                f"{self.query_y.shape if isinstance(self.query_y, torch.Tensor) else self.query_y}, single_eval_pos={self.single_eval_pos}, " \
                f" padding_mask={self.observed}")

    def __len__(self):
        return self.x.shape[1]

    @property
    def observed(self):
        return (~self.padding_mask).sum(dim=1)


class PaddedBatch:
    """
    A batch of data with padding for the train set.
    """

    def __init__(self, x, y, query_x, query_y, single_eval_pos, max_train_len, padding_mask,
                 target_idx=0):
        self.x = x
        self.y = y
        self.query_x = query_x
        self.query_y = query_y
        self.single_eval_pos = single_eval_pos
        self.max_train_len = max_train_len
        self.padding_mask = padding_mask
        self.target_idx = target_idx

    @property
    def target_task(self):
        return MyBatch(
            x=self.x[:, self.target_idx:self.target_idx + 1, :],
            y=self.y[:, self.target_idx:self.target_idx + 1],
            query_x=self.query_x[:, self.target_idx:self.target_idx + 1, :],
            query_y=self.query_y[:, self.target_idx:self.target_idx + 1],
            single_eval_pos=self.x.shape[0],
            padding_mask=self.padding_mask[self.target_idx:self.target_idx + 1, :],
        )

    @property
    def related_tasks(self):
        indices = [i for i in range(self.x.shape[1]) if i != self.target_idx]
        return MyBatch(x=self.x[:, indices, :], y=self.y[:, indices],
                       single_eval_pos=self.x.shape[0],
                       padding_mask=self.padding_mask[indices, :], )


def parse_batch_for_padded_train_data(batch, target_idx=0):
    # Notice, that the batch.x tensor is regular, but the single_eval_pos is a list
    x = batch.x
    y = batch.y
    batch_size = x.shape[1]
    max_train_len = max(batch.single_eval_pos)
    dim = x.shape[2]

    # Prepare padded train set and mask
    train_data = []
    train_y = []
    padding_masks = []

    for batch_idx, train_len in enumerate(batch.single_eval_pos):
        # Extract the train part for this batch element
        data_x = x[:train_len, batch_idx, :]  # (train_len, dim)
        data_y = y[:train_len, batch_idx]
        # Pad to max_train_len
        pad_len = max_train_len - train_len
        if pad_len > 0:
            pad = torch.zeros(pad_len, dim, dtype=x.dtype, device=x.device)
            data_x = torch.cat([data_x, pad], dim=0)
            data_y = torch.cat([data_y, pad[:, 0]], dim=0)
        train_data.append(data_x)
        train_y.append(data_y)
        # Mask: True for padding, False for real data
        mask = torch.cat([
            torch.zeros(train_len, dtype=torch.bool, device=x.device),
            torch.ones(pad_len, dtype=torch.bool, device=x.device)
        ])
        padding_masks.append(mask)

    # Stack into tensors
    train_tensor = torch.stack(train_data, dim=1)  # (max_train_len, batch, dim)
    train_tensor_y = torch.stack(train_y, dim=1)
    src_key_padding_mask = torch.stack(padding_masks, dim=0)  # (batch, max_train_len)

    query_x = x[max_train_len:, :, :]  # (n_query, batch, dim)
    query_y = y[max_train_len:, :]  # (n_query, batch)

    # extend the src_key_padding_mask to the query points
    # query_mask = torch.zeros(query_x.shape[0], batch_size, dtype=torch.bool, device=x.device)
    # src_key_padding_mask = torch.cat([src_key_padding_mask, query_mask], dim=0)

    return PaddedBatch(
        x=train_tensor,
        y=train_tensor_y,
        query_y=query_y,
        query_x=query_x,
        single_eval_pos=batch.single_eval_pos,
        max_train_len=max_train_len,
        padding_mask=src_key_padding_mask,
        target_idx=target_idx
    )


if __name__ == "__main__":
    import src.ifBO_main.ifbo as ifbo_main

    root = Path(__file__).parents[2]

    torch.cuda.empty_cache()
    # ftpfn = FTPFN(target_path=root / "src/.model", version="0.0.1")

    device = torch.device("cpu")
    ftpfn = ifbo_main.surrogate.FTPFN(version="0.0.1", device=device)
    from src.dataset.taskprior import MetaTaskPriorSameProblem

    dataset = MetaTaskPriorSameProblem(
        dim_hyperparameters=3,
        n_fidelities=100,
        seq_len=1000,
    )

    single_eval_pos = [200, 300]
    batch = dataset.sample_batch(
        n_tasks=2,
        single_eval_pos=single_eval_pos,
        alphas=[0.5, 0.5],
    )

    model = ftpfn.model

    padded_batch = parse_batch_for_padded_train_data(batch)
    train_tensor = padded_batch.x
    train_tensor_y = padded_batch.y
    query_x = padded_batch.query_x
    src_key_padding_mask = padded_batch.padding_mask

    assert train_tensor.shape == (max(single_eval_pos), 2, 3)
    assert src_key_padding_mask.shape == (2, max(single_eval_pos))
    assert src_key_padding_mask.shape[1] == train_tensor.shape[0], \
        "only ever in reference to the cat of train_tensor and query_x!"
    output = model.forward(
        (torch.cat([train_tensor, query_x], dim=0).to(device), train_tensor_y.to(device)),
        single_eval_pos=max(single_eval_pos),
        src_key_padding_mask=src_key_padding_mask.to(device),
    )

    # Testing that the forward is indeed the same with padding:
    outputs_unpadded = []

    for b in range(train_tensor.shape[1]):  # iterate over batch dimension
        # Find valid (unpadded) positions for this batch item
        valid_idx = ~src_key_padding_mask[b]  # bool mask: True for valid positions
        n_valid = valid_idx.sum().item()

        # Select only the valid (unpadded) part for this batch item
        x_unpadded = train_tensor[valid_idx, b, :].unsqueeze(1)  # (T_valid, 1, D)
        y_unpadded = train_tensor_y[valid_idx, b].unsqueeze(1)  # (T_valid, 1)
        query_x_b = query_x[:, b:b + 1, :]  # (T_query, 1, D)

        # Forward pass for this batch item (without padding)
        output_b = model.forward(
            (torch.cat([x_unpadded, query_x_b], dim=0), y_unpadded),
            single_eval_pos=n_valid,
            # src_key_padding_mask=None  # No mask needed, as no padding
        )
        outputs_unpadded.append(output_b)

        # assert torch.allclose(
        #     output[:, b][n_valid],
        #     output_b[:, b][:n_valid],
        #     atol=1e-6
        # ), f"Mismatch in batch item {b} for valid positions"


    # Stack outputs to shape (T_query, B)
    outputs_unpadded = torch.cat(outputs_unpadded, dim=1)

    assert torch.allclose(output, outputs_unpadded, atol=1e-6), \
        "Padded and unpadded forward passes do not match!"


    output = ftpfn.batch_query_forward(
        x_train=train_tensor.to(device),
        y_train=train_tensor_y.to(device),
        x_query=query_x.to(device).repeat(5, 1, 1),
        single_eval_pos=max(single_eval_pos),
        padding_mask=src_key_padding_mask.to(device)
    )





    print()
