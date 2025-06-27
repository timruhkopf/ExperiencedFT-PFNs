from pfns4bo.utils import to_tensor
import torch
import numpy as np
import pandas as pd
from pfns4bo_utils import general_power_transform
from torch import nn
import sys
sys.path.append("../..")
from src.model.pfnimputation import PFNPriorImputation
from src.model.calc_reliability import calc_imputed_linalg_reliability
from src.model.decay import constant_exponential as decay_fn
from src.model.mixing import argmin as mixture_fn
from types import SimpleNamespace
from hpob_handler import HPOBHandler
import logging
logger = logging.getLogger(__name__)

class ourTransformerBOMethod(nn.Module):
    def __init__(self, model, related_task_data, device='cpu:0', fit_encoder = None, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = model.criterion
        self.device = device
        self.kwargs = kwargs
        self.fit_encoder = fit_encoder
        x_task_context = to_tensor(np.stack([ item["X"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)
        y_task_context =  to_tensor(np.stack([ item["y"]   for k, item in related_task_data.items()], axis=1)).to(torch.float32)  
        y_task_context =  self.label_transforms(y_task_context, **self.kwargs)
        padding_mask = torch.zeros(x_task_context.size(1), x_task_context.size(0), dtype=torch.bool).to(device)

        self.related_task_data = SimpleNamespace(x=x_task_context, y=y_task_context, padding_mask=padding_mask)

        self.pfnimputation = PFNPriorImputation(
            self,
            self.criterion, #logger
            None,
            decay_fn,
            mixture_fn,
            self.related_task_data,
            min_context_size=1,
            imputation_mode='mean',
            reliability_fn=calc_imputed_linalg_reliability,
            device=device,
        )

    @torch.no_grad()
    def observe_and_suggest(self, X_obs, y_obs, X_pen, return_actual_ei=False):
        # X_obs is a numpy array of shape (n_samples, n_features)
        # y_obs is a numpy array of shape (n_samples,), between 0 and 1
        # X_pen is a numpy array of shape (n_samples_left, n_features)
        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."
        X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
        
        y_obs = to_tensor(y_obs, device=self.device).to(torch.float32)
        y_obs =  self.label_transforms(y_obs, **self.kwargs).squeeze()
        X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)

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

        return  self.model(
            (style,
            x_full,
            y_full),
            single_eval_pos=single_eval_pos,
        )


def get_meta_data_from_hpob_hdlr(search_space_id,seeds,  seed, time_horizon):
    hpob_hdlr = HPOBHandler(root_dir="./HPO_B/hpob-data/", mode="v3-train-augmented")
    dataset_ids = hpob_hdlr.get_datasets(search_space_id)
    meta_data = {}
    for dataset_id in dataset_ids:
        init_ids = hpob_hdlr.bo_initializations[search_space_id][dataset_id][seed]
        X = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["X"])
        y = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["y"])
        y = hpob_hdlr.normalize(y).squeeze()

        rng = np.random.default_rng(seeds.index(seed))
        all_ids = np.arange(len(X))
        rest = np.setdiff1d(all_ids, init_ids)
        meta_configurations = np.concatenate([init_ids, rng.choice(rest, time_horizon - len(init_ids), replace=False)])
        

        meta_data[dataset_id] = {
            "X": X[meta_configurations],
            "y": y[meta_configurations],
            "meta_configurations": meta_configurations
        }
    return meta_data


def get_meta_data(benchmark_name, hpob_hdlr, search_space_id, down_stream_dataset_id, dataset_ids,seeds,  seed):
    meta_data = {}
    for dataset_id in dataset_ids:
        if dataset_id != down_stream_dataset_id:
            X = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["X"])
            y = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["y"])
            y = hpob_hdlr.normalize(y).squeeze()
            meta_data[dataset_id] = {
                "X": X,
                "y": y,
            }
    return meta_data


def get_meta_data_of_method_name(benchmark_name, base_method_name, hpob_hdlr, search_space_id, down_stream_dataset_id, dataset_ids,seeds,  seed):
    result_folder = "./results/"
    meta_data = {}
    for dataset_id in dataset_ids:
        if dataset_id != down_stream_dataset_id:
            df = pd.read_csv(result_folder + benchmark_name + "_" + base_method_name + "_" + search_space_id +"_"+ dataset_id + ".csv")
            meta_configurations = df.sort_values(by=['seed', 'iteration'])['configuration'].values.reshape(len(seeds), -1)
            meta_configuration = meta_configurations[seeds.index(seed)]
            
            X = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["X"])[meta_configuration]
            y = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["y"])[meta_configuration]
            y = hpob_hdlr.normalize(y).squeeze()
            meta_data[dataset_id] = {
                "X": X,
                "y": y,
                "meta_configurations": meta_configuration
            }
    return meta_data