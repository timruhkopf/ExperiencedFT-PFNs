import math


def k_folds(train_ids, k=5):
    """Splits train_ids into k non-overlapping folds."""
    n = len(train_ids)

    fold_size = int(math.ceil(n / k))
    return [train_ids[i*fold_size:(i+1)*fold_size] for i in range(k)]


def folds_of_size(train_ids, size=5, drop=True):
    """Split train_ids into non overlapping folds of at most size"""

    if size <= 0:
        return [train_ids]

    if drop:
        n = len(train_ids) // size
    else:
        n =  int(math.ceil(len(train_ids) / size))

    return [train_ids[i*size:(i+1)*size] for i in range(n)]