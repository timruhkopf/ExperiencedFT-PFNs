import numpy as np
import pandas as pd
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification, Objective
from benchmarks.SimpleSynth.task_sampler import get_batch_test
import json
import time
import torch
from typing import Dict, List, Union, Optional
import random


class SimpleSynth:
    def __init__(
        self,
        seed: Optional[int] = None,
        start_task: int = 8,
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

    def get_meta_data(self, task_ids=[3, 4, 5, 7, 9]):
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

        print(X[current_evaluations[-1]])
    # negate to recover accuracy
    min_regret_history += [min(y).item()] * (max_evaluations - i - 1)

    return min_regret_history, opt_time