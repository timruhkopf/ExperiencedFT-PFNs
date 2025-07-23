from pfns4bo.utils import to_tensor
import torch
import numpy as np
import pandas as pd
from .pfns4bo_utils import general_power_transform
from torch import nn
import sys
sys.path.append("../../")
from src.model.pfnimputation import PFNPriorImputation
from src.model.mixing import CVMixtureStrategy
from src.model.calc_reliability import calc_imputed_linalg_reliability
# from src.model.decay import constant_exponential as decay_fn
# from src.model.mixing import argmin as mixture_fn
from types import SimpleNamespace
import logging
logger = logging.getLogger(__name__)

class MPFNs4BO(nn.Module):
    def __init__(self, model, search_space, related_task_data, validation_task_data = None, device='cpu:0', fit_encoder = None, apply_power_transform =False, input_power_transform=False, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = model.criterion
        self.device = device
        self.kwargs = kwargs
        self.fit_encoder = fit_encoder
        self.search_space = search_space
        self.apply_power_transform = apply_power_transform
        self.input_power_transform = input_power_transform
        self.input_power_transform_eps = 0.0

        """Meta-learning on meta-data, corresponds to the meta-learning part in Algorithm 1."""
        converted_meta_data = dict()
        max_length = 0
        # for task_uid, evaluations in related_task_data.items():
        #     X = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations])
        #     Y = -np.array([e.objectives["loss"] for e in evaluations]).reshape(-1) # return to maximization (performance)
        #     max_length = max(max_length, len(Y))
        #     converted_meta_data[task_uid] = {"X": X, "y": Y}
        #     print(Y.shape, X.shape, task_uid, Y.max(), Y.min())
        for task_uid, evaluations in related_task_data.items():
            X = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations])
            Y = -np.array([e.objectives["loss"] for e in evaluations]).reshape(-1) # return to maximization (performance)
            if task_uid in validation_task_data:
                evaluations_val = validation_task_data[task_uid]
                X_val = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations_val])
                if self.input_power_transform :
                    X_val = self.power_transforms(X_val, **self.kwargs).squeeze()
                Y_val = -np.array([e.objectives["loss"] for e in evaluations_val]).reshape(-1) # return to maximization (performance)
                Y_val = (Y_val-np.min(Y_val))/(np.max(Y_val)-np.min(Y_val))
                if self.apply_power_transform:
                    Y_val = self.power_transforms(Y_val, **self.kwargs).squeeze()
                X = np.concatenate([X, X_val], axis=0)
                Y = np.concatenate([Y, Y_val], axis=0)
            max_length = max(max_length, len(Y))
            converted_meta_data[task_uid] = {"X": X, "y": Y}


        x_task_context =[]
        y_task_context = []
        padding_mask = []
        for k, item in converted_meta_data.items():
            x_task_context.append(np.concatenate([item["X"], np.zeros((max_length - item["X"].shape[0] , item["X"].shape[1]))]))
            y_task_context.append(np.concatenate( [item["y"] , np.zeros((max_length - item["y"].shape[0]))]))  
            padding_mask.append(np.concatenate([np.zeros(item["y"].shape[0]),  np.ones(max_length- item["y"].shape[0])]  ))
        x_task_context = to_tensor(np.stack(x_task_context, axis=1)).to(torch.float32).to(device)
        y_task_context =  to_tensor(np.stack(y_task_context, axis=1)).to(torch.float32).to(device)
        padding_mask =  to_tensor(np.stack(padding_mask, axis=1)).to(torch.bool).to(device).T 

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

    @torch.no_grad()
    def observe_and_suggest(self, X_obs, y_obs, X_pen, return_actual_ei=False, minimize=True):
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)
        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."
        if minimize:
            y_obs = to_tensor(1 - y_obs, device=self.device).to(torch.float32).view(-1) # data are normalized between 0 and 1
        else:
            y_obs = to_tensor(y_obs, device=self.device).to(torch.float32).view(-1)
        X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
        X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)

        if self.apply_power_transform:
            y_obs = self.power_transforms(y_obs, **self.kwargs).squeeze()
        if self.input_power_transform:
            X_obs = general_power_transform(X_obs, X_obs, self.input_power_transform_eps)
            X_pen = general_power_transform(X_obs, X_pen, self.input_power_transform_eps)

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
        acq_mask = acq_values.max() == acq_values
        possible_next = torch.arange(len(X_pen))[acq_mask]
        if len(possible_next) == 0:
            possible_next = torch.arange(len(X_pen))
        r = possible_next[torch.randperm(len(possible_next))[0]].cpu().item()

        if return_actual_ei:
            return r, acq_values
        else:
            return r

    def power_transforms(
        self,
        y_given,
        apply_power_transform=True,
        power_transform_eps=0.0,
        unsafe_power_transform=False,
    ):
        if isinstance(y_given, np.ndarray):
            y_given = torch.tensor(y_given, device=self.device)
        if len(y_given.shape) == 1:
            y_given = y_given.unsqueeze(1)
        if apply_power_transform:
            y_given = general_power_transform(y_given, y_given, power_transform_eps, less_safe=unsafe_power_transform)
        return y_given

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
                x_full_masked = x_full[:, batch_index:batch_index+1, :][~src_x_padding_mask]
                y_full_masked = y_full[:, batch_index:batch_index+1][~src_y_padding_mask]
                single_eval_pos_masked = single_eval_pos - int(src_y_padding_mask.sum())
                if single_eval_pos_masked > 1000:
                    idx = torch.randperm(single_eval_pos_masked)[:1000]
                    y_full_masked = y_full_masked[idx]

                    idx = torch.cat([
                        idx,
                        torch.arange(single_eval_pos_masked, x_full_masked.shape[0])
                    ])
                    x_full_masked = x_full_masked[idx] 
                    
                    single_eval_pos_masked = 1000

                res = self.model(
                    (style,
                        x_full_masked,
                        y_full_masked
                    ),
                    single_eval_pos=single_eval_pos_masked
                )
                results.append(res)
            return torch.cat(results, dim=1)