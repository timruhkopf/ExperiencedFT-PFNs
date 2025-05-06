def k_folds(train_ids, k=5):
    """Splits train_ids into k non-overlapping folds."""
    n = len(train_ids)

    fold_size = n // k
    return [train_ids[i*fold_size:(i+1)*fold_size] for i in range(k)]
