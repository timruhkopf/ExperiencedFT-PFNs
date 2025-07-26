import torch
import scipy
import math
from sklearn.preprocessing import power_transform, PowerTransformer
import numpy as np


def general_power_transform(x_train, x_apply, eps=0.0, less_safe=False):
    if isinstance(x_train, np.ndarray) or isinstance(x_apply, np.ndarray):
        x_train = torch.tensor(x_train)
        x_apply = torch.tensor(x_apply)
    if eps > 0:
        try:
            pt = PowerTransformer(method='box-cox')
            pt.fit(x_train.cpu()+eps)
            x_out = torch.tensor(pt.transform(x_apply.cpu()+eps), dtype=x_apply.dtype, device=x_apply.device)
        except Exception as e:
            print(e)
            x_out = x_apply - x_train.mean(0)
    else:
        pt = PowerTransformer(method='yeo-johnson')
        if not less_safe and (x_train.std() > 1_000 or x_train.mean().abs() > 1_000):
            x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            x_train = (x_train - x_train.mean(0)) / x_train.std(0)
            print('inputs are LAARGEe, normalizing them')
        try:
            pt.fit(x_train.cpu().double())
        except Exception as e:
            print('caught this errrr', e)
            if less_safe:
                x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            else:
                x_train = x_train - x_train.mean(0)
                x_apply = x_apply - x_train.mean(0)
            try:
                pt.fit(x_train.cpu().double())
            except Exception as e2:
                print('caught this err2', e2)
                x_out = x_apply - x_train.mean(0)
                return x_out
        try:
            x_out = torch.tensor(pt.transform(x_apply.cpu()), dtype=x_apply.dtype, device=x_apply.device)
        except Exception as e3:
                print('transform err3', e3)
                x_out = x_apply - x_train.mean(0)
                return x_out
    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        print('WARNING: power transform failed')
        #print(f"{x_train=} and {x_apply=}")
        x_out = x_apply - x_train.mean(0)
    return x_out
