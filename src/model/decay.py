import torch


def constant_exponential(n_target, lambda_=0.001, constant=0):
    effective_n = torch.maximum(torch.tensor(n_target - constant, dtype=torch.float32),
                                torch.tensor(0.0))
    return 1 - torch.exp(-lambda_ * effective_n)


