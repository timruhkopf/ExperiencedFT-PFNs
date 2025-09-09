from pfns4bo.utils import to_tensor
import torch
import numpy as np
import pandas as pd
from .pfns4bo_utils import general_power_transform
from torch import nn
import sys
sys.path.append("../../")
from types import SimpleNamespace
import logging
logger = logging.getLogger(__name__)
from functools import partial, update_wrapper

from src.model.abstractmodel import AbstractModel

from src.model.initial_design.max_acq_initial_design import MaxAcqInitialDesign, RepeatedMaxAcqInitialDesign
from src.model.strategies.error_model import ErrorModelStrategies
from src.model.strategies.meta_context import JointBatchedMetaContextStrategy, JointMetaContextStrategy, SplitMetaContextStrategy
from src.model.weights.simple import MeanWeights
from src.model.weights.past_surprise import PastSurpriseWeights
from src.model.components.imputor import PriorImputer

import copy 

class MetaPFNs4BO(nn.Module):
    def __init__(self, model, search_space, related_task_data, validation_task_data = [], device='cpu:0', fit_encoder = None, configuration = {}, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = model.criterion
        self.device = device
        self.kwargs = kwargs
        self.fit_encoder = fit_encoder
        self.search_space = search_space
        self.configuration = configuration
        self.apply_power_transform = configuration.get("apply_power_transform", True)
        self.input_power_transform = configuration.get("input_power_transform", False)
        self.input_power_transform_eps = configuration.get("input_power_transform_eps", 0.0)
        self.acquisition_function_type = configuration.get("acquisition_function_type", "ei")
        self.flippable =  configuration.get("flippable", False)

        self.related_task_data = self.prepare_meta_data(related_task_data, validation_task_data)
        self.related_task_data_copy = copy.deepcopy(self.related_task_data)


        initial_design = configuration.get("initial_design", {"type": "maxacq", "params": {"size":10}})
        if initial_design["type"] == "maxacq":
            params = initial_design.get("params", {})
            self.initial_design = MaxAcqInitialDesign(**params)
        elif initial_design["type"] == "repeated_maxacq":
            params = initial_design.get("params", {})
            self.initial_design = RepeatedMaxAcqInitialDesign(**params)
        else:
            raise ValueError(f"Unknown initial design type: {initial_design['type']}")

        strategy = configuration.get("strategy", {"type": "joint-context", "imputer": {"type": "prior-imputer", "params": {"imputation_mode": "median"}}})
        strategy_imputer = strategy.get("imputer", {})
        if strategy_imputer["type"] == "prior-imputer":
            params = strategy_imputer.get("params", {})
            self.strategy_imputer = PriorImputer(**params)
        else:
            self.strategy_imputer = None

        if strategy["type"] == "error-model":
            params = strategy.get("params", {})
            self.strategy = ErrorModelStrategies(**params)
        elif strategy["type"] == "joint-context":
            params = strategy.get("params", {})
            self.strategy = JointMetaContextStrategy(imputer=self.strategy_imputer, **params)
        elif strategy["type"] == "joint-batched-context":
            params = strategy.get("params", {})
            self.strategy = JointBatchedMetaContextStrategy(imputer=self.strategy_imputer, **params)
        elif strategy["type"] == "split-context":
            params = strategy.get("params", {})
            self.strategy = SplitMetaContextStrategy(imputer=self.strategy_imputer, **params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy['type']}")

        weights = configuration.get("weights", {"type": "mean-weights"})
        if weights["type"] == "mean-weights":
            params = weights.get("params", {})
            self.weights = MeanWeights(**params)
        elif weights["type"] == "past-surprise":
            params = weights.get("params", {})
            self.weights = PastSurpriseWeights(**params)
        else:
            raise ValueError(f"Unknown weights type: {weights['type']}")

        imputer = configuration.get("imputer", {"type": "prior-imputer", "params": {"imputation_mode": "median"}})
        if imputer["type"] == "prior-imputer":
            params = imputer.get("params", {})
            self.imputer = PriorImputer(**params)
        else:
            raise ValueError(f"Unknown imputer type: {imputer['type']}")

        self.ppfn = AbstractModel(
            initial_design = self.initial_design,
            strategy = self.strategy,
            flippable = self.flippable,
            logger = None,
            model = self,
            device = self.device,
            related_task_data = self.related_task_data,
            weights=self.weights,  # weight strategy
            imputer=self.imputer,
            callbacks=(),
            contender_bonus=None,
            acquisition_function_type=self.acquisition_function_type,
        )

    def prepare_meta_data(self, related_task_data, validation_task_data = []):
        """Meta-learning on meta-data, corresponds to the meta-learning part in Algorithm 1."""
        converted_meta_data = dict()
        max_length = 0
        for task_uid, evaluations in related_task_data.items():
            X = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations])
            Y = -self.normalize(np.array([e.objectives["loss"] for e in evaluations]).reshape(-1)) # return to maximization (performance)
            if self.input_power_transform :
                X = self.power_transforms(X, **self.kwargs).squeeze()
            if self.apply_power_transform:
                Y = self.power_transforms(Y, **self.kwargs).squeeze()
            if task_uid in validation_task_data:
                evaluations_val = validation_task_data[task_uid]
                X_val = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations_val])
                if self.input_power_transform :
                    X_val = self.power_transforms(X_val, **self.kwargs).squeeze()
                Y_val = - self.normalize(np.array([e.objectives["loss"] for e in evaluations_val]).reshape(-1)) # return to maximization (performance)
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
        x_task_context = to_tensor(np.stack(x_task_context, axis=1)).to(torch.float32).to(self.device)
        y_task_context =  to_tensor(np.stack(y_task_context, axis=1)).to(torch.float32).to(self.device)
        padding_mask =  to_tensor(np.stack(padding_mask, axis=1)).to(torch.bool).to(self.device).T 

        return  SimpleNamespace(x=x_task_context, y=y_task_context, padding_mask=padding_mask)

    def normalize(self, y):
        return (y-np.min(y))/(np.max(y)-np.min(y)+ 1e-8)

    @torch.no_grad()
    def observe_and_suggest(self, X_obs, y_obs, X_pen, return_actual_ei=False, minimize=True):
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)
        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."
        if minimize:
            y_obs = to_tensor(-y_obs, device=self.device).to(torch.float32).view(-1) # data are normalized between 0 and 1
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
            self.related_task_data.x = w(self.related_task_data_copy.x)


        if self.acquisition_function_type == "ei":
            acquisition_function = self.ppfn.get_ei
        elif self.acquisition_function_type == "pi":
            acquisition_function = self.ppfn.get_pi
        else:
            raise ValueError(f"Unknown acquisition_function_type: {self.acquisition_function_type}")

        acq_values = acquisition_function(
            X_pen,
            y_obs.max(),
            x_train=X_obs,
            y_train=y_obs,
            minimize=False,
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
        style = None,
        max_num_samples=500,):
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
                if single_eval_pos_masked > max_num_samples:
                    idx = torch.randperm(single_eval_pos_masked)[:max_num_samples]
                    y_full_masked = y_full_masked[idx]

                    idx = torch.cat([
                        idx,
                        torch.arange(single_eval_pos_masked, x_full_masked.shape[0])
                    ])
                    x_full_masked = x_full_masked[idx] 
                    
                    single_eval_pos_masked = max_num_samples

                res = self.model(
                    (style,
                        x_full_masked,
                        y_full_masked
                    ),
                    single_eval_pos=single_eval_pos_masked
                )
                results.append(res)
            return torch.cat(results, dim=1)