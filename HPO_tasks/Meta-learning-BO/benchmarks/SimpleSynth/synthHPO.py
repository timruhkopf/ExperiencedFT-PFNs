import numpy as np
import pandas as pd
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective
import json
import time
import torch
from typing import Dict, List, Union, Optional
import random

import numpy as np

import numpy as np

class MetaHPOBenchmark:
    def __init__(self, dim=16, n_tasks=50, seed=0, p_noninformative=0.2, p_redundant=0.2):
        """
        Meta-HPO synthetic benchmark generator with shifts, non-informative, and redundant dims.
        
        Args:
            dim (int): Total dimensionality of the search space.
            n_tasks (int): Number of meta-tasks.
            seed (int): Random seed for reproducibility.
            p_noninformative (float): fraction of dimensions that are pure noise.
            p_redundant (float): fraction of dimensions that are redundant copies of relevant ones.
        """
        self.dim = dim
        self.n_tasks = n_tasks
        self.rng = np.random.RandomState(seed)
        self.p_noninformative = p_noninformative
        self.p_redundant = p_redundant
        
        self.base_task = self._make_base_task()  # shared Gaussian mixture
        self.tasks = [self._make_task(i) for i in range(n_tasks)]
    
    def _make_base_task(self):
        """Create a base Gaussian mixture landscape shared across tasks."""
        rng = self.rng
        n_components = rng.randint(4, 7)  # number of modes
        means = rng.rand(n_components, self.dim)
        covs = rng.uniform(0.05, 0.2, size=(n_components, self.dim))
        weights = rng.dirichlet(np.ones(n_components))
        return {"n_components": n_components, "means": means, "covs": covs, "weights": weights}
    
    def _make_task(self, task_id):
        """Create task-specific variations (shift + relevant/non-informative/redundant dims)."""
        rng = np.random.RandomState(task_id)
        
        # Relevant dims
        n_relevant = rng.randint(5, self.dim)  # at least 5 relevant
        relevant_dims = rng.choice(self.dim, n_relevant, replace=False)
        
        # Non-informative dims (pure noise, independent of function)
        available_dims = [d for d in range(self.dim) if d not in relevant_dims]
        n_noninf = min(int(self.p_noninformative * self.dim), len(available_dims))
        #n_noninf = int(self.p_noninformative * self.dim)
        noninformative_dims = rng.choice(
            [d for d in range(self.dim) if d not in relevant_dims],
            size=n_noninf,
            replace=False
        ) if n_noninf > 0 else []
        
        # Redundant dims (copies of some relevant ones)
        available_dims = [d for d in range(self.dim) if d not in relevant_dims]
        n_redundant = min(int(self.p_redundant * self.dim), len(available_dims))
        redundant_map = {}
        if n_redundant > 0 and len(relevant_dims) > 0:
            redundant_dims = rng.choice(
                [d for d in range(self.dim) if d not in relevant_dims],
                size=n_redundant,
                replace=False
            )
            for rd in redundant_dims:
                redundant_map[rd] = rng.choice(relevant_dims)
        else:
            redundant_dims = []
        
        # Task-specific shift
        shift = rng.uniform(-0.2, 0.2, size=self.dim)
        
        return {
            "relevant_dims": relevant_dims,
            "noninformative_dims": noninformative_dims,
            "redundant_map": redundant_map,
            "shift": shift,
        }
    
    def evaluate(self, x, task_id):
        """
        Evaluate the benchmark function at point x for task_id.
        
        Args:
            x (array-like): shape (d,) or (n, d) in [0,1]^dim
            task_id (int): which task to evaluate (0..n_tasks-1)
            
        Returns:
            float or np.ndarray: function value(s)
        """
        x = np.atleast_2d(x)
        task = self.tasks[task_id]
        base = self.base_task
        
        means, covs, weights = base["means"], base["covs"], base["weights"]
        relevant_dims, shift = task["relevant_dims"], task["shift"]
        noninf_dims, redundant_map = task["noninformative_dims"], task["redundant_map"]
        
        # Apply shift
        x_shifted = x + shift
        
        # Handle redundant dims (replace with their source relevant dim)
        if redundant_map:
            for rd, src in redundant_map.items():
                x_shifted[:, rd] = x_shifted[:, src]
        
        # Handle non-informative dims (inject Gaussian noise)
        if len(noninf_dims) > 0:
            noise = np.random.normal(0, 0.05, size=(x_shifted.shape[0], len(noninf_dims)))
            x_shifted[:, noninf_dims] = noise
        
        # Use only relevant dims
        x_rel = x_shifted[:, relevant_dims]
        means_rel = means[:, relevant_dims]
        covs_rel = covs[:, relevant_dims]
        
        scores = []
        for i in range(len(weights)):
            diff = (x_rel - means_rel[i]) / covs_rel[i]
            scores.append(weights[i] * np.exp(-0.5 * np.sum(diff**2, axis=1)))
        
        result = -np.log(np.sum(scores, axis=0) + 1e-12)  # minimization objective
        return result if len(result) > 1 else result[0]

    def get_batch_test(self, spt_size=4, task=0, number_points=200):

        X_spt = torch.rand(spt_size, self.dim)
        X_qry = torch.rand(number_points, self.dim)

        y_spt = self.evaluate(X_spt, task_id=task)
        y_qry = self.evaluate(X_qry, task_id=task)

        return X_spt, y_spt, X_qry, y_qry


class SynthHPO:
    def __init__(
        self,
        seed: Optional[int] = None,
        start_task: int = 0,
        initializations: int = 3,
        n_tasks: int = 50,
        dim: int = 15,
        p_noninformative: float = 0.2,
        p_redundant: float = 0.2,
    ):
        self.num_tasks = n_tasks
        self.dim = dim
        self.seed = seed
        if seed is not None:
            seed_num = int(''.join(filter(str.isdigit, seed)))
            np.random.seed(seed_num)
            torch.manual_seed(seed_num)
            random.seed(seed_num)

        self.benchmark = MetaHPOBenchmark(dim=dim, n_tasks=n_tasks, seed=seed_num if seed is not None else 0, p_noninformative=p_noninformative, p_redundant=p_redundant)
        self.X_spt, self.y_spt, self.X_qry, self.y_qry = self.benchmark.get_batch_test(spt_size=initializations, task=start_task)
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

    def get_meta_data(self, task_ids= range(0, 49)):
        meta_data: Dict[Union[str, int, List[Evaluation]]] = dict()

        for task_id in task_ids:
            meta_data[task_id] = []
            self.X_spt, self.y_spt, self.X_qry, self.y_qry = self.benchmark.get_batch_test(spt_size=self.initializations, task=task_id)
            xs = np.concatenate([self.X_spt, self.X_qry])
            ys = np.concatenate([self.y_spt, self.y_qry])
            for x, y in zip(xs, ys):
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

def normalize_y(y):
    y = (y - np.min(y)) / (np.max(y) - np.min(y))  # normalize to 0 - 1
    return y

def run_optimization_loop(
    benchmark,
    test_seed,
    optimizer,
    max_evaluations: int,
    run_on_meta_train = False
):
    X = np.asarray(benchmark.benchmark_data["X"])
    y = np.asarray(benchmark.benchmark_data["y"])
    y = normalize_y(y)
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