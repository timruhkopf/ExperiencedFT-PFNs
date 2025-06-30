import torch
import matplotlib.pyplot as plt
import numpy as np

class ScaledFunctionWrapper:
    def __init__(self, func, y_transform = None, X_transforms = None, n_samples=10000):
        """
        Wrap a BoTorch synthetic function with input and output scaling.
        
        Args:
            func: A BoTorch synthetic function (e.g., Branin()).
            n_samples: Number of samples for estimating output bounds.
        """
        self.func = func
        self.dim = func.dim
        self.bounds = [(0, 1)   for i in range(self.dim )]

        self.y_transform = (lambda x: x) if y_transform is None else y_transform

        if X_transforms is None:
            self.X_transforms = [lambda x: x] * self.dim
        else:
            assert len(X_transforms) == self.dim, "X_transforms must match input dimensions"
            self.X_transforms = X_transforms

        # Estimate output range using random sampling
        self.y_min, self.y_max = self.estimate_output_bounds(n_samples)


    def scale_to_bounds(self, X_unit):
        """Convert inputs from [0, 1]^d to original bounds."""
        lower = self.func.bounds[:, 0]
        upper = self.func.bounds[:, 1]
        X_scaled = lower + (upper - lower) * X_unit
        for i, transform in enumerate(self.X_transforms):
            X_scaled[i] = transform(X_scaled[i])
        return X_scaled

    def normalize_outputs(self, y):
        """Normalize output to [0, 1] using estimated bounds."""
        y_trans = self.y_transform(y)
        return (y_trans - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def estimate_output_bounds(self, n_samples):
        """Estimate the output range by sampling the function."""
        X_unit = torch.rand(n_samples, self.dim)  # Uniform in [0, 1]^d
        X = self.scale_to_bounds(X_unit)
        Y = self.y_transform(self.func(X))
        return Y.min(), Y.max()

    def __call__(self, X_unit):
        """
        Evaluate the wrapped function on scaled inputs in [0, 1]^d.
        
        Args:
            X_unit: Tensor of shape [n, d] with entries in [0, 1].
            
        Returns:
            Normalized function values in [0, 1].
        """
        X_scaled = self.scale_to_bounds(X_unit)
        Y = self.func(X_scaled)
        return self.normalize_outputs(Y)