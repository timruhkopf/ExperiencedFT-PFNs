import numpy as np

class RandomSearch:
    def __init__(self, bounds):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)

    def observe(self, x, y):
        pass

    def suggest(self):
        """Suggest the next point to evaluate by sampling."""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
