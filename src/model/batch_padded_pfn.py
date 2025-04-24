from pathlib import Path

import torch


def parse_batch_for_padded_train_data(batch):
    # Notice, that the batch.x tensor is regular, but the single_eval_pos is a list
    x = batch.x
    y = batch.y
    batch_size = x.shape[1]
    max_train_len = max(single_eval_pos)
    dim = x.shape[2]

    # Prepare padded train set and mask
    train_data = []
    train_y = []
    padding_masks = []

    for batch_idx, train_len in enumerate(single_eval_pos):
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

    # extend the src_key_padding_mask to the query points
    # query_mask = torch.zeros(query_x.shape[0], batch_size, dtype=torch.bool, device=x.device)
    # src_key_padding_mask = torch.cat([src_key_padding_mask, query_mask], dim=0)

    return train_tensor, train_tensor_y, query_x, src_key_padding_mask


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

    single_eval_pos = [200, 500]
    batch = dataset.sample_batch(
        n_tasks=2,
        single_eval_pos=single_eval_pos,
        alphas=[0.5, 0.5],
    )

    model = ftpfn.model
    train_tensor, train_tensor_y, query_x, src_key_padding_mask \
        = parse_batch_for_padded_train_data(batch)



    assert train_tensor.shape == (max(single_eval_pos), 2, 3)
    assert src_key_padding_mask.shape == ( 2, max(single_eval_pos))
    assert src_key_padding_mask.shape[1] == train_tensor.shape[0] , \
        "only ever in reference to the cat of train_tensor and query_x!"
    output = model.forward(
        (torch.cat([train_tensor, query_x], dim=0).to(device),train_tensor_y.to(device)),
        single_eval_pos=max(single_eval_pos),
        src_key_padding_mask=src_key_padding_mask.to(device),
    )

    output = ftpfn.batch_query_forward(
        x_train=train_tensor.to(device),
        y_train=train_tensor_y.to(device),
        x_query=query_x.to(device).repeat(5, 1, 1),
        single_eval_pos=max(single_eval_pos),
        padding_mask=src_key_padding_mask.to(device)
    )
    print(ftpfn)