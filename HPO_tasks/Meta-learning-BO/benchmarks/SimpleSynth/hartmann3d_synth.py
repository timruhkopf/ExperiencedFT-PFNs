import numpy as np
import pandas as pd
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective
import json
import time
import torch
from typing import Dict, List, Union, Optional
import random


def hartmann3d(x, task=0):
    """
    Hartmann 3-dimensional test function.
    The function is usually evaluated on the unit cube x_i ∈ (0, 1), for all i = 1, 2, 3.
    Global minimum: f(x*) = -3.86278 at x* ≈ [0.114614, 0.555649, 0.852547]
    """
    alphas = [
    [1.0, 1.2, 3.0, 3.2],         # Original (default)
    [0.5, 0.5, 0.5, 0.5],         # Uniform low weight
    [2.0, 2.0, 2.0, 2.0],         # Uniform high weight
    [1.0, 2.0, 3.0, 4.0],         # Linearly increasing
    [4.0, 3.0, 2.0, 1.0],         # Linearly decreasing
    [1.0, 0.0, 0.0, 0.0],         # Only first term active
    [0.0, 0.0, 0.0, 1.0],         # Only last term active
    [0.1, 10.0, 0.1, 10.0],       # Alternating small and large
    [5.0, 1.0, 0.2, 2.5],         # Arbitrary varying weights
    [0.3, 0.7, 1.1, 0.9],         # Smooth varying low weights
    ]

    alpha = alphas[task]  # Choose the first set of weights for the function

    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])

    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])

    if isinstance(x, list):
        x = np.array(x)
    x = np.atleast_2d(x)
    
    assert x.shape[1] == 3, "Input must have shape (n_samples, 3)"

    # Compute (X - P)^2 * A, shape: (n_samples, 4)
    diff = x[:, np.newaxis, :] - P[np.newaxis, :, :]         # (n, 4, 3)
    prod = A[np.newaxis, :, :] * diff**2                     # (n, 4, 3)
    exp_term = np.exp(-np.sum(prod, axis=2))                 # (n, 4)
    result = -np.dot(exp_term, alpha)                        # (n,)

    return result if x.shape[0] > 1 else result[0]


def get_batch_test(spt_size=4, task=0, step=0.1):

    # Generate 3D query points in the unit cube [0, 1]^3
    grid = np.arange(0, 1, step)
    mesh = np.meshgrid(grid, grid, grid)
    X_qry = torch.Tensor(np.stack([m.flatten() for m in mesh], axis=1))
    ix = [random.sample(grid.tolist(), spt_size) for _ in range(3)]
    ix = np.stack(ix, axis=1)
    X_spt = torch.Tensor(ix)

    y_spt = hartmann3d(X_spt, task=task)
    y_qry = hartmann3d(X_qry, task=task)

    return X_spt, y_spt, X_qry, y_qry


class Hartmann3D:
    def __init__(
        self,
        seed: Optional[int] = None,
        start_task: int = 0,
        initializations: int = 3,
    ):
        
        self.seed = seed
        if seed is not None:
            seed_num = int(''.join(filter(str.isdigit, seed)))
            np.random.seed(seed_num)
            torch.manual_seed(seed_num)
            random.seed(seed_num)
        self.X_spt, self.y_spt, self.X_qry, self.y_qry = get_batch_test(spt_size=initializations, task=start_task)
        self.init_ids = list(range(len(self.X_spt)))
        self.data = {
            "X": np.concatenate([self.X_spt, self.X_qry]),
            "y": np.concatenate([self.y_spt, self.y_qry]),
        }
        
        self.initializations = initializations
        

    @property
    def search_space(self):
        space = ps.ParameterSpace()
        X = self.data["X"][0]
        search_space_dims = len(X)
        for n in range(search_space_dims):
            # default parameter type is "uniform_float"
            space._parameters[f"x_{n}"] = {
                "parameter": ps.ContinuousParameter(
                    name=f"x_{n}",
                    bounds=(0, 1.),
                    transformation=None,
                ),
                "condition": ps.Condition(),
            }
        return space

    @property
    def bo_initializations(self):
        return {self.seed: self.init_ids}

    @property
    def benchmark_data(self):
        return self.data

    def get_meta_data(self, task_ids= range(1, 5)):
        meta_data: Dict[Union[str, int, List[Evaluation]]] = dict()

        for task_id in task_ids:
            meta_data[task_id] = []
            self.X_spt, self.y_spt, self.X_qry, self.y_qry = get_batch_test(spt_size=self.initializations, task=task_id)
            xs = np.concatenate([self.X_spt, self.X_qry])
            ys = np.concatenate([self.y_spt, self.y_qry])
            for x, y in zip(xs, -ys):
                config = {}
                for n, x_n in enumerate(x):
                    config[f'x_{n}'] = x_n
                eval_spec = EvaluationSpecification(configuration=config)
                evaluation = eval_spec.create_evaluation(
                    objectives={"loss": y.item()},
                    user_info={
                        "task_uid": task_id,
                        "cost": None
                    },
                )
                meta_data[task_id].append(evaluation)
        return meta_data

def run_optimization_loop(
    benchmark,
    test_seed,
    optimizer,
    max_evaluations: int,
    run_on_meta_train = False
):
    X = np.asarray(benchmark.benchmark_data["X"])
    y = np.asarray(benchmark.benchmark_data["y"])
    y = 1 - y
    #print(f"Normalized y: {y.min()} to {y.max()}")
    #print(X.shape, y.shape)

    data_size = len(X)
    # indices of pending evaluations
    pending_evaluations = list(range(data_size))
    current_evaluations = []

    init_ids = benchmark.bo_initializations[test_seed]
    for i in range(len(init_ids)):
        idx = init_ids[i]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)

    # NOTE: change max to min
    min_regret_history = [np.min(y[current_evaluations])]
    opt_time = []
    for i in range(max_evaluations):
        # take the acquistion values from the pending evaluations
        start_time = time.time()
        idx = optimizer.observe_and_suggest(
            X[current_evaluations], y[current_evaluations], X[pending_evaluations]
        )
        end_time = time.time()
        opt_time.append(end_time - start_time)
        idx = pending_evaluations[idx]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)
        min_regret_history.append(np.min(y[current_evaluations]))

        if min(y) in min_regret_history:
            break

        print(f"Iteration {i + 1}/{max_evaluations}, "
              f"Current best: {min_regret_history[-1]:.4f}, "
              f"Time taken: {opt_time[-1]:.4f} seconds")

    # negate to recover accuracy
    min_regret_history += [min(y).item()] * (max_evaluations - i - 1)

    return min_regret_history, opt_time