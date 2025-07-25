from pfns4bo.utils import to_tensor
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings = warnings

import pandas as pd
from sklearn.preprocessing import PowerTransformer
from torch import nn
import sys
sys.path.append("../..")
from src.model.pfnimputation import PFNPriorImputation
from src.model.mixing import CVMixtureStrategy
from functools import partial

#from src.model.calc_reliability import calc_imputed_linalg_reliability
# from src.model.decay import constant_exponential as decay_fn
# from src.model.mixing import argmin as mixture_fn
from types import SimpleNamespace
import contextlib
import logging
logger = logging.getLogger(__name__)


class ourPFNs4BO(nn.Module):
    def __init__(self, model, related_task_data,  bounds , device='cpu:0', n_candidates=1000, initial_points = 5, fit_encoder=None, **kwargs):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        n_candidates: number of random samples for acquisition optimization
        """
        super().__init__()

        self.model = model
        self.criterion = model.criterion
        self.device = device


        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.n_candidates = n_candidates
        self.X = []
        self.y = []

        self.kwargs = kwargs
        self.fit_encoder = fit_encoder
        self.initial_points = initial_points
        

        x_task_context = to_tensor(np.stack([ item["X"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)
        y_task_context =  to_tensor(np.stack([ item["y"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)  
        #y_task_context =  self.label_transforms(y_task_context, **self.kwargs)
        padding_mask = torch.zeros(x_task_context.size(1), x_task_context.size(0), dtype=torch.bool).to(device)

        self.related_task_data = SimpleNamespace(x=x_task_context, y=y_task_context, padding_mask=padding_mask)

        self.pfnimputation = PFNPriorImputation(
            model = self,
            criterion = self.criterion, #logger
            logger = None,
            mixture_strategy = CVMixtureStrategy,
            related_task_data = self.related_task_data,
            min_context_size =1,
            imputation_mode ='mean',
            device=device,
        )

        self.reliability_scores = torch.zeros(len(related_task_data.items())).to(device) 

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

        acq_values = self.pfnimputation.get_pi(
                X_pen,
                y_obs.max(),
                x_train=X_obs,
                y_train=y_obs,
                minimize= False,
            ).squeeze()

        self.reliability_scores = self.pfnimputation.reliability_scores

        best_idx = np.argmax(acq_values)
        return candidates[best_idx]

    def forward(
        self,
        src: tuple,
        single_eval_pos: int | None = None,
        src_key_padding_mask=None,
        style = None,):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        x_full , y_full = src
        if style is not None:
            if callable(style):
                style = style()
            if isinstance(style, torch.Tensor):
                style = style.to(x_full.device)
            else:
                style = torch.tensor(style, device=x_full.device).view(1, 1).repeat(x_full.shape[1], 1)

        if src_key_padding_mask is None:
            return self.model(
            (style,
            x_full,
            y_full),
            single_eval_pos=single_eval_pos,
        )
        else:
            # PFNs4BO does not support src_key_padding_mask! 
            # We need to do it manually
            results = []
            for batch_index in range(src_key_padding_mask.shape[0]): #batch size
                src_x_padding_mask =  torch.nn.functional.pad(src_key_padding_mask[batch_index], (0, x_full.shape[0] - src_key_padding_mask.shape[1]), value=False)
                src_y_padding_mask = src_key_padding_mask[batch_index]
                res = self.model(
                (style,
                x_full[:, batch_index:batch_index+1, :][~src_x_padding_mask] ,
                y_full[:, batch_index:batch_index+1][~src_y_padding_mask]
                ),
                single_eval_pos= single_eval_pos - int(src_y_padding_mask.sum())
                )
                results.append(res.clone().detach())
            return torch.cat(results, dim=1)

    def label_transforms(
        self,
        y_given,
        apply_power_transform=True,
        power_transform_eps=.0,
        unsafe_power_transform=False,
    ):
        if len(y_given.shape) == 1:
            y_given = y_given.unsqueeze(1)
        if apply_power_transform:
            y_given = general_power_transform(y_given, y_given, power_transform_eps, less_safe=unsafe_power_transform)
        return y_given

def general_power_transform(x_train, x_apply, eps, less_safe=False):
    if eps > 0:
        try:
            pt = PowerTransformer(method='box-cox')
            pt.fit(x_train.cpu()+eps)
            x_out = torch.tensor(pt.transform(x_apply.cpu()+eps), dtype=x_apply.dtype, device=x_apply.device)
        except ValueError as e:
            print(e)
            x_out = x_apply - x_train.mean(0)
    else:
        pt = PowerTransformer(method='yeo-johnson')
        if not less_safe and (x_train.std() > 1_000 or x_train.mean().abs() > 1_000):
            x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            x_train = (x_train - x_train.mean(0)) / x_train.std(0)
            print('inputs are LAARGEe, normalizing them')
        try:
            pt.fit(x_train.cpu().double())
        except ValueError as e:
            print('caught this errrr', e)
            if less_safe:
                x_train = (x_train - x_train.mean(0)) / x_train.std(0)
                x_apply = (x_apply - x_train.mean(0)) / x_train.std(0)
            else:
                x_train = x_train - x_train.mean(0)
                x_apply = x_apply - x_train.mean(0)
            pt.fit(x_train.cpu().double())
        x_out = torch.tensor(pt.transform(x_apply.cpu()), dtype=x_apply.dtype, device=x_apply.device)
    if torch.isnan(x_out).any() or torch.isinf(x_out).any():
        print('WARNING: power transform failed')
        print(f"{x_train=} and {x_apply=}")
        x_out = x_apply - x_train.mean(0)
    return x_out




def get_meta_data_of_method_name(meta_data_lists, seeds,  seed):
    result_folder = "./results/"
    meta_data = {}
    for i, (function_name, transform_id,base_method_name) in enumerate(meta_data_lists):
        df = pd.read_csv(result_folder + function_name + "_" + base_method_name  + ".csv")
        df = df[(df['transform_id'] == int(transform_id)) & (df['seed'] == seeds.index(seed))]
        x_array = np.stack(df['x'].str.strip('[]').str.split().apply(lambda lst: list(map(float, lst))))
        meta_data[i] = {
            "X": x_array,
            "y": df['y'].values
        }
    return meta_data


class ourPFNs4BO_discrete(nn.Module):
    def __init__(self, model, related_task_data, transformation_type = None, device='cpu:0', fit_encoder=None, **kwargs):
        """
        bounds: List of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        n_candidates: number of random samples for acquisition optimization
        """
        super().__init__()

        self.model = model
        self.criterion = model.criterion
        self.device = device
        self.X = []
        self.y = []

        self.kwargs = kwargs
        self.fit_encoder = fit_encoder
        self.transformation_type = transformation_type
        self.evaluated_indices = []  # To keep track of already evaluated candidates

        x_task_context = to_tensor(np.stack([ item["X"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)
        y_task_context =  to_tensor(np.stack([ item["y"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)  
        #y_task_context =  self.label_transforms(y_task_context, **self.kwargs)
        padding_mask = torch.zeros(x_task_context.size(1), x_task_context.size(0), dtype=torch.bool).to(device)

        self.related_task_data = SimpleNamespace(x=x_task_context, y=y_task_context, padding_mask=padding_mask)

        self.pfnimputation = PFNPriorImputation(
            model = self,
            criterion = self.criterion, #logger
            logger = None,
            mixture_strategy = partial(CVMixtureStrategy, transformation_type=self.transformation_type),
            related_task_data = self.related_task_data,
            min_context_size =1,
            imputation_mode ='mean',
            device=device,

        )

        self.reliability_scores = torch.zeros(len(related_task_data.items())).to(device) 

    def observe(self, x, y):
        """Add an observation of x (1D array-like) with target y (float)."""
        self.X.append(np.array(x))
        self.y.append(np.array(y))

    def suggest(self,candidates,  return_actual_ei=True):
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

        acq_values = self.pfnimputation.get_pi(
                X_pen,
                y_obs.max(),
                x_train=X_obs,
                y_train=y_obs,
                minimize= False,
            ).squeeze()

        self.reliability_scores = self.pfnimputation.reliability_scores
        print(f"Reliability scores: {self.reliability_scores}")

        if self.evaluated_indices:
            idx_tensor = torch.tensor(self.evaluated_indices, dtype=torch.long, device=acq_values.device)
            acq_values[idx_tensor] = 0

        acq_values = acq_values.detach().cpu().numpy()
        best_idx = np.argmax(acq_values)
        self.evaluated_indices.append(best_idx)
        return best_idx, acq_values

    def forward(
        self,
        src: tuple,
        single_eval_pos: int | None = None,
        src_key_padding_mask=None,
        style = None,):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        x_full , y_full = src
        if style is not None:
            if callable(style):
                style = style()
            if isinstance(style, torch.Tensor):
                style = style.to(x_full.device)
            else:
                style = torch.tensor(style, device=x_full.device).view(1, 1).repeat(x_full.shape[1], 1)

        if src_key_padding_mask is None:
            return self.model(
            (style,
            x_full,
            y_full),
            single_eval_pos=single_eval_pos,
        )
        else:
            # PFNs4BO does not support src_key_padding_mask! 
            # We need to do it manually
            results = []
            for batch_index in range(src_key_padding_mask.shape[0]): #batch size
                src_x_padding_mask =  torch.nn.functional.pad(src_key_padding_mask[batch_index], (0, x_full.shape[0] - src_key_padding_mask.shape[1]), value=False)
                src_y_padding_mask = src_key_padding_mask[batch_index]
                res = self.model(
                (style,
                x_full[:, batch_index:batch_index+1, :][~src_x_padding_mask] ,
                y_full[:, batch_index:batch_index+1][~src_y_padding_mask]
                ),
                single_eval_pos= single_eval_pos - int(src_y_padding_mask.sum())
                )
                results.append(res.clone().detach())
            return torch.cat(results, dim=1)

    def label_transforms(
        self,
        y_given,
        apply_power_transform=True,
        power_transform_eps=.0,
        unsafe_power_transform=False,
    ):
        if len(y_given.shape) == 1:
            y_given = y_given.unsqueeze(1)
        if apply_power_transform:
            y_given = general_power_transform(y_given, y_given, power_transform_eps, less_safe=unsafe_power_transform)
        return y_given