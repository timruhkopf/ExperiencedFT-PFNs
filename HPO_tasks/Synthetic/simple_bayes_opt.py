import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer:
    def __init__(self, bounds, kernel=None, alpha=1e-6, n_candidates=1000, initial_points = 5):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        kernel: scikit-learn kernel, default is Matern
        n_candidates: number of random samples for acquisition optimization
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.n_candidates = n_candidates
        self.X = []
        self.y = []

        if kernel is None:
            kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
        self.initial_points = initial_points

    def observe(self, x, y):
        """Add an observation of x (1D array-like) with target y (float)."""
        self.X.append(np.array(x))
        self.y.append(y)
        self.gp.fit(np.array(self.X), np.array(self.y))

    def suggest(self, acquisition="ucb", kappa=2.0):
        """Suggest the next point to evaluate by sampling."""
        if len(self.X) < self.initial_points:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Sample random candidates
        candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_candidates, self.dim))
        mu, sigma = self.gp.predict(candidates, return_std=True)

        if acquisition == "ucb":
            scores = mu + kappa * sigma
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

        best_idx = np.argmax(scores)
        return candidates[best_idx]
