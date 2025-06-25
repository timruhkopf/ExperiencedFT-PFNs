from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
import torch


class HeboOptimizer:
    def __init__(self, search_space_dim, maximize=True):
        """
        :param space_bounds: List of (lower, upper) tuples for each dimension
        :param maximize: Whether to maximize the objective (HEBO always minimizes, so we flip sign if maximizing)
        """
        self.maximize = maximize
        self.dim = search_space_dim

        # Define HEBO search space
        self.space = DesignSpace().parse([
            {'name': f'x{i}', 'type': 'num', 'lb': 0, 'ub': 1} for i in range(self.dim)])

        cfg = {
                    "lr": 0.0001,
                    "num_epochs": 100,
                    "verbose": True,
                    "noise_lb": 8e-4,
                    "pred_likeli": False,
                }
        self.opt = HEBO(self.space) #, model_config=cfg,)
        self.first_run = True

    def observe_and_suggest(self, X_obs=None, y_obs=None, X_pen=None):
        y = np.array(y_obs)
        X = np.array(X_obs)
        if self.maximize:
            y = -y # HEBO minimizes

        if self.first_run:
            suggestion = self.opt.suggest(n_suggestions=5)
            X_dict = pd.DataFrame([{f'x{i}': x[i] for i in range(self.dim)} for x in X])
            self.opt.observe(X_dict, y)
            self.first_run = False
        else:
            x_dict = {f'x{i}': X[-1, i] for i in range(self.dim)}
            self.opt.observe(pd.DataFrame([x_dict]), np.asarray([y[-1]]))


        # Penalized evaluation (return index of best in X_pen)
        X_pen = np.array(X_pen)
        X_pen_dict = pd.DataFrame([{f'x{i}': x[i] for i in range(self.dim)} for x in X_pen])
        suggestion = self.opt.suggest(n_suggestions=1)

        matching_indices = X_pen_dict.index[X_pen_dict.apply(lambda row: row.equals(suggestion.iloc[0]), axis=1)]
        match_index = matching_indices[0] if not matching_indices.empty else None
        print(suggestion)
        print(match_index)
        return match_index