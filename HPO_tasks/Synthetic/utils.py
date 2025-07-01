import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

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
        self.epsilon = 1e-5  # Small value to avoid division by zero

        self.y_transform = (lambda x: x) if y_transform is None else y_transform

        if X_transforms is None:
            self.X_transforms = [lambda x: x] * self.dim
        else:
            assert len(X_transforms) == self.dim, "X_transforms must match input dimensions"
            self.X_transforms = X_transforms

        # Estimate output range using random sampling
        self.y_min, self.y_max = self.estimate_output_bounds(n_samples)
        print(f"Estimated output bounds: {self.y_min:.4f}, {self.y_max:.4f}")


    def scale_to_bounds(self, X_unit):
        """Convert inputs from [0, 1]^d to original bounds."""
        lower = self.func.bounds[0, :]
        upper = self.func.bounds[1, :]
        X_transformed = np.zeros_like(X_unit)
        for i, transform in enumerate(self.X_transforms):
            X_transformed[i] = transform(X_unit[i])
        X_scaled = lower + (upper - lower) * X_transformed
        return X_scaled

    def normalize_outputs(self, y, maximize=True):
        """Normalize output to [0, 1] using estimated bounds."""
        y_trans = self.y_transform(y)
        y_norm = (y_trans - self.y_min) / (self.y_max - self.y_min + self.epsilon)
        return  1 - torch.clamp(y_norm, 0.0, 1.0) if maximize  else torch.clamp(y_norm, 0.0, 1.0)

    def estimate_output_bounds(self, n_samples):
        """Estimate the output range by sampling the function."""
        binary_combinations = torch.tensor(list(itertools.product([0.0, 1.0], repeat=self.dim)))
        X_augmented = torch.cat([torch.rand(n_samples, self.dim), binary_combinations], dim=0)

        X = self.scale_to_bounds(X_augmented)
        Y = self.y_transform(self.func(X))

        optimal_value = self.func.optimal_value
        if optimal_value is not None:
            Y = torch.cat((Y, torch.tensor([optimal_value], dtype=Y.dtype)))

        return Y.min()* (1 -self.epsilon),  Y.max() * (1 + self.epsilon)  # Add small epsilon to avoid division by zero

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
        y =  self.normalize_outputs(Y)
        return y