import numpy as np
import torch
from pfns4bo.scripts.acquisition_functions import general_acq_function, to_tensor
import contextlib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings = warnings


class PFNs4BO:
    def __init__(self, model,  bounds, acq_f=general_acq_function, device='cpu:0', n_candidates=1000, initial_points = 5, fit_encoder=None, **kwargs):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        n_candidates: number of random samples for acquisition optimization
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.n_candidates = n_candidates
        self.X = []
        self.y = []

        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder
        self.initial_points = initial_points

    def observe(self, x, y):
        """Add an observation of x (1D array-like) with target y (float)."""
        self.X.append(np.array(x))
        self.y.append(y)

    def suggest(self, return_actual_ei=False):
        """Suggest the next point to evaluate by sampling."""
        if len(self.X)<self.initial_points:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Sample random candidates
        candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_candidates, self.dim))

        X_obs = to_tensor(self.X, device=self.device).to(torch.float32)
        y_obs = to_tensor(self.y, device=self.device).to(torch.float32).view(-1)
        X_pen = to_tensor(candidates, device=self.device).to(torch.float32)

        #print(f"Observations shape: {X_obs.shape}, Targets shape: {y_obs.shape}, Candidates shape: {X_pen.shape}")

        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)

        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
            acq_values = self.acq_function(self.model, X_obs, y_obs,
                                           X_pen, return_actual_ei=return_actual_ei, **self.kwargs).cpu().clone()  # bool array

        best_idx = np.argmax(acq_values)
        return candidates[best_idx]







class PFNs4BO_discrete:
    def __init__(self, model, acq_f=general_acq_function, device='cpu:0', fit_encoder=None, **kwargs):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        n_candidates: number of random samples for acquisition optimization
        """
        self.X = []
        self.y = []
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder
        self.evaluated_indices = []  # To keep track of already evaluated candidates

    def observe(self, x, y):
        """Add an observation of x (1D array-like) with target y (float)."""
        self.X.append(np.array(x))
        self.y.append(y)

    def suggest(self, candidates, return_actual_ei=True):
        """Suggest the next point to evaluate by sampling."""
        X_obs = to_tensor(self.X, device=self.device).to(torch.float32)
        y_obs = to_tensor(self.y, device=self.device).to(torch.float32).view(-1)
        X_pen = to_tensor(candidates, device=self.device).to(torch.float32)

        #print(f"Observations shape: {X_obs.shape}, Targets shape: {y_obs.shape}, Candidates shape: {X_pen.shape}")

        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)

        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
            acq_values = self.acq_function(self.model, X_obs, y_obs,
                                           X_pen, return_actual_ei=return_actual_ei, **self.kwargs).cpu().clone()  # bool array

        if self.evaluated_indices:
            idx_tensor = torch.tensor(self.evaluated_indices, dtype=torch.long, device=acq_values.device)
            acq_values[idx_tensor] = 0

        acq_values = acq_values.detach().cpu().numpy()
        best_idx = np.argmax(acq_values)
        self.evaluated_indices.append(best_idx)
        return best_idx, acq_values


