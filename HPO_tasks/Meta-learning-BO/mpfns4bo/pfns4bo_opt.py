import contextlib
import torch
from pfns4bo.utils import to_tensor
from pfns4bo.scripts.tune_input_warping import fit_input_warping
from .pfns4bo_utils import general_acq_function
import numpy as np

class PFNs4BO:
    def __init__(self, model, acq_f=general_acq_function, device='cpu:0', fit_encoder=None, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.acq_function = acq_f
        self.fit_encoder = fit_encoder


    @torch.no_grad()
    def observe_and_suggest(self, X_obs, y_obs, X_pen, return_actual_ei=False, minimize=True):
        # assert X_pen is not None
        # assumptions about X_obs and X_pen:
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)
        if minimize:
            y_obs = to_tensor(1 - y_obs, device=self.device).to(torch.float32).view(-1) # data are normalized between 0 and 1
        else:
            y_obs = to_tensor(y_obs, device=self.device).to(torch.float32).view(-1)
        X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
        X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)

        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)

        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)

        with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
            acq_values = self.acq_function(self.model, X_obs, y_obs,
                                           X_pen, apply_power_transform = False, acq_function='pi', **self.kwargs).cpu().clone()  # bool array
            acq_mask = acq_values.max() == acq_values
        possible_next = torch.arange(len(X_pen))[acq_mask]
        if len(possible_next) == 0:
            possible_next = torch.arange(len(X_pen))

        r = possible_next[torch.randperm(len(possible_next))[0]].cpu().item()

        #print(f"Suggesting point {r} with acquisition value {acq_values}, y_obs: {y_obs.max(), y_obs[-1]}")
        # run on one dataset 
        
        
        if return_actual_ei:
            return r, acq_values
        else:
            return r







