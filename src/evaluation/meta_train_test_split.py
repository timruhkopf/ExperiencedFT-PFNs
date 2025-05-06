import math


def k_folds(train_ids, k=5):
    """Splits train_ids into k non-overlapping folds."""
    n = len(train_ids)

    fold_size = int(math.ceil(n / k))
    return [train_ids[i*fold_size:(i+1)*fold_size] for i in range(k)]


def folds_of_size(train_ids, size=5):
    """Split train_ids into non overlapping folds of at most size"""
    n = len(train_ids)
    return [train_ids[i*size:(i+1)*size] for i in range(size)]