from hebo.design_space.design_space import DesignSpace
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
        self.opt = HEBO(self.space)

    def observe_and_suggest(self, X_obs=None, y_obs=None, X_pen=None):
        for x, y in zip(X_obs, y_obs):
            if self.maximize:
                y = -y  # HEBO minimizes
            x_dict = {f'x{i}': x[i] for i in range(self.dim)}
            self.opt.observe(pd.DataFrame([x_dict]), np.array([[y]]))

        # Penalized evaluation (return index of best in X_pen)
        print(dir(self.opt))
        X_pen = np.array(X_pen)
        X_pen_dict = pd.DataFrame([{f'x{i}': x[i] for i in range(self.dim)} for x in X_pen])
        mu, var = self.opt.model.predict(X_pen_dict)
        acq = -mu.flatten() if self.maximize else mu.flatten()  # back to maximization if needed
        return np.argmax(acq)