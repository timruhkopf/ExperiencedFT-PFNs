from pfns4bo.utils import to_tensor
import torch
import numpy as np
import pandas as pd
from .pfns4bo_utils import general_power_transform
from torch import nn
import sys
sys.path.append("../../")
from src.model.ppfn import PPFN
from types import SimpleNamespace
import logging
logger = logging.getLogger(__name__)
from functools import partial
import copy 
from pfns4bo import transformer
from pfns4bo import bar_distribution
import contextlib


def mixing_strategy(src, src_meta = None, mix_rate =0.5):
    return mix_rate * src_meta.unsqueeze(1)  + (1 - mix_rate) * src


def f_support_attn(src_left, src_ ,self_attn,  src_meta = None, mix_rate =0.1):
    src_meta_attn = self_attn(src_, src_meta.unsqueeze(1), src_meta.unsqueeze(1))[0]
    return mix_rate * src_meta_attn  + (1 - mix_rate) * src_left


class AggregatedPPFNs4BO(nn.Module):
    def __init__(self, model, search_space, related_task_data, validation_task_data = [], device='cpu:0', fit_encoder = None, apply_power_transform =False, apply_power_transform_ds =False, input_power_transform=False, **kwargs):
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
        self.apply_power_transform_ds = apply_power_transform_ds
        self.acq_function_type = 'ei'

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
        x_task_context = to_tensor(np.stack(x_task_context, axis=1)).to(torch.float32).to(device)
        y_task_context =  to_tensor(np.stack(y_task_context, axis=1)).to(torch.float32).to(device)
        padding_mask =  to_tensor(np.stack(padding_mask, axis=1)).to(torch.bool).to(device).T 

        self.related_task_data = SimpleNamespace(x=x_task_context, y=y_task_context, padding_mask=padding_mask)
        self.related_task_data_copy = copy.deepcopy(self.related_task_data)
        self.attention_outputs = None
        self.hidden_state_outputs = None
        self.target_logits = None
        self.meta_logits = None

    def normalize(self, y):
        return (y-np.min(y))/(np.max(y)-np.min(y)+ 1e-8)

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

    @torch.no_grad()
    def observe_and_suggest(self, X_observed, y_observed, X_pending, return_actual_ei=False, minimize=True):
        if minimize:
            y_obs = to_tensor(-y_observed, device=self.device).to(torch.float32).view(-1) # data are normalized between 0 and 1
        else:
            y_obs = to_tensor(y_observed, device=self.device).to(torch.float32).view(-1)
        X_obs = to_tensor(X_observed, device=self.device).to(torch.float32)
        X_pen = to_tensor(X_pending, device=self.device).to(torch.float32)

        assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

        self.model.to(self.device)


        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask
        num_related = related_context_y.shape[1]


        if self.fit_encoder is not None:
            w = self.fit_encoder(self.model, X_obs, y_obs)
            X_obs = w(X_obs)
            X_pen = w(X_pen)
            related_context_x = w(related_context_x)

        imputed_logits = self.batch_forward(
            (
                torch.cat([
                    related_context_x,
                    X_obs.unsqueeze(1).repeat(1, num_related, 1),
                    X_obs.unsqueeze(1).repeat(1, num_related, 1),
                    X_pen.unsqueeze(1).repeat(1, num_related, 1),
                ], dim=0),
                related_context_y
            ),
            single_eval_pos=related_context_x.shape[0],
            src_key_padding_mask=padding_mask
            )
        self.meta_logits = imputed_logits.cpu().clone()

        # attention_outputs = torch.stack(self.attention_outputs)[:,:,related_context_x.shape[0]:].squeeze()  # Todo padding_mask should be used
        # aggreated_atention_outputs = torch.mean(attention_outputs, dim=0)
        # layers_info = []
        # for layer in range(len(aggreated_atention_outputs)):
        #     if layer < 6:
        #         layers_info.append(( layer ,  {"f_attn": partial(mixing_strategy, src_meta = aggreated_atention_outputs[layer], mix_rate = 0.2) } ))
        #     else:
        #         layers_info.append(( layer ,  {} ))
        hidden_state_outputs = torch.stack(self.hidden_state_outputs)[:,:,related_context_x.shape[0]:].squeeze()  # Todo padding_mask should be used
        aggreated_outputs = torch.mean(hidden_state_outputs, dim=0)
        layers_info = []
        for layer in range(len(aggreated_outputs)):
            if layer < 6:
                layers_info.append(( layer ,  {"f_hidden_state": partial(mixing_strategy, src_meta = aggreated_outputs[layer]) } ))
            else:
                layers_info.append(( layer ,  {} ))

        with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
            acq_values = self.general_acq_function( X_obs, y_obs,
                                           X_pen, apply_power_transform=self.apply_power_transform, input_power_transform=self.input_power_transform, acq_function=self.acq_function_type, layers_info = layers_info, **self.kwargs).cpu().clone()  # bool array
            acq_mask = acq_values.max() == acq_values
        possible_next = torch.arange(len(X_pen))[acq_mask]
        if len(possible_next) == 0:
            possible_next = torch.arange(len(X_pen))

        r = possible_next[torch.randperm(len(possible_next))[0]].cpu().item()
        
        if return_actual_ei:
            return r, acq_values
        else:
            return r
        
    def batch_forward(
        self,
        src: tuple,
        single_eval_pos: int | None = None,
        src_key_padding_mask=None,
        style = None,
        max_num_samples=1024,):
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
            results = self.model(
            (style,
            x_full,
            y_full),
            single_eval_pos=single_eval_pos,)
            self.attention_outputs = self.model.get_attention_outputs()
            self.hidden_state_outputs = self.model.get_hidden_state_outputs()
            return results
        
        else:
            # PFNs4BO does not support src_key_padding_mask! 
            # We need to do it manually
            results = []
            attention_outputs = []
            hidden_state_outputs = []
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
                attention_outputs.append(self.model.get_attention_outputs())
                hidden_state_outputs.append(self.model.get_hidden_state_outputs())
            self.attention_outputs = attention_outputs
            self.hidden_state_outputs = hidden_state_outputs
            return torch.cat(results, dim=1)
    
    #@torch.inference_mode()
    def general_acq_function(self, x_given, y_given, x_eval, apply_power_transform=True,
                        rand_sample=False, znormalize=False, pre_normalize=False, pre_znormalize=False, predicted_mean_fbest=False,
                        input_znormalize=False, max_dataset_size=10_000, remove_features_with_one_value_only=False,
                        return_actual_ei=False, acq_function='ei', ucb_rest_prob=.05, ensemble_log_dims=False,
                        ensemble_type='mean_probs', # in ('mean_probs', 'max_acq')
                        input_power_transform=False, power_transform_eps=.0, input_power_transform_eps=.0,
                        input_rank_transform=False, ensemble_input_rank_transform=False,
                        ensemble_power_transform=False, ensemble_feature_rotation=False,
                        style=None, outlier_stretching_interval=0.0, verbose=False, unsafe_power_transform=False, layers_info = None
                            ):
        """
        Differences to HEBO:
            - The noise can't be set in the same way, as it depends on the tuning of HPs via VI.
            - Log EI and PI are always used directly instead of using the approximation.

        This is a stochastic function, relying on torch.randn

        :param model:
        :param x_given: torch.Tensor of shape (N, D)
        :param y_given: torch.Tensor of shape (N, 1) or (N,)
        :param x_eval: torch.Tensor of shape (M, D)
        :param kappa:
        :param eps:
        :return:
        """
        assert ensemble_type in ('mean_probs', 'max_acq')
        if rand_sample is not False \
            and (len(x_given) == 0 or
                ((1 + x_given.shape[1] if rand_sample is None else max(2, rand_sample)) > x_given.shape[0])):
            print('rando')
            return torch.zeros_like(x_eval[:,0]) #torch.randperm(x_eval.shape[0])[0]
        y_given = y_given.reshape(-1)
        assert len(y_given) == len(x_given)
        if apply_power_transform:
            if pre_normalize:
                y_normed = y_given / y_given.std()
                if not torch.isinf(y_normed).any() and not torch.isnan(y_normed).any():
                    y_given = y_normed
            elif pre_znormalize:
                y_znormed = (y_given - y_given.mean()) / y_given.std()
                if not torch.isinf(y_znormed).any() and not torch.isnan(y_znormed).any():
                    y_given = y_znormed
            y_given = general_power_transform(y_given.unsqueeze(1), y_given.unsqueeze(1), power_transform_eps, less_safe=unsafe_power_transform).squeeze(1)
            if verbose:
                print(f"{y_given=}")
            #y_given = torch.tensor(power_transform(y_given.cpu().unsqueeze(1), method='yeo-johnson', standardize=znormalize), device=y_given.device, dtype=y_given.dtype,).squeeze(1)
        y_given_std = torch.tensor(1., device=y_given.device, dtype=y_given.dtype)
        if znormalize and not apply_power_transform:
            if len(y_given) > 1:
                y_given_std = y_given.std()
            y_given_mean = y_given.mean()
            y_given = (y_given - y_given_mean) / y_given_std

        if remove_features_with_one_value_only:
            x_all = torch.cat([x_given, x_eval], dim=0)
            only_one_value_feature = torch.tensor([len(torch.unique(x_all[:,i])) for i in range(x_all.shape[1])]) == 1
            x_given = x_given[:,~only_one_value_feature]
            x_eval = x_eval[:,~only_one_value_feature]

        if outlier_stretching_interval > 0.:
            tx = torch.cat([x_given, x_eval], dim=0)
            m = outlier_stretching_interval
            eps = 1e-10
            small_values = (tx < m) & (tx > 0.)
            tx[small_values] = m * (torch.log(tx[small_values] + eps) - math.log(eps)) / (math.log(m + eps) - math.log(eps))

            large_values = (tx > 1. - m) & (tx < 1.)
            tx[large_values] = 1. - m * (torch.log(1 - tx[large_values] + eps) - math.log(eps)) / (
                        math.log(m + eps) - math.log(eps))
            x_given = tx[:len(x_given)]
            x_eval = tx[len(x_given):]

        if input_znormalize: # implementation that relies on the test set, too...
            std = x_given.std(dim=0)
            std[std == 0.] = 1.
            mean = x_given.mean(dim=0)
            x_given = (x_given - mean) / std
            x_eval = (x_eval - mean) / std

        if input_power_transform:
            x_given = general_power_transform(x_given, x_given, input_power_transform_eps)
            x_eval = general_power_transform(x_given, x_eval, input_power_transform_eps)

        if input_rank_transform is True or input_rank_transform == 'full': # uses test set x statistics...
            x_all = torch.cat((x_given,x_eval), dim=0)
            for feature_dim in range(x_all.shape[-1]):
                uniques = torch.sort(torch.unique(x_all[..., feature_dim])).values
                x_eval[...,feature_dim] = torch.searchsorted(uniques,x_eval[..., feature_dim]).float() / (len(uniques)-1)
                x_given[...,feature_dim] = torch.searchsorted(uniques,x_given[..., feature_dim]).float() / (len(uniques)-1)
        elif input_rank_transform is False:
            pass
        elif input_rank_transform == 'train':
            x_given = rank_transform(x_given, x_given)
            x_eval = rank_transform(x_given, x_eval)
        elif input_rank_transform.startswith('train'):
            likelihood = float(input_rank_transform.split('_')[-1])
            if torch.rand(1).item() < likelihood:
                print('rank transform')
                x_given = rank_transform(x_given, x_given)
                x_eval = rank_transform(x_given, x_eval)
        else:
            raise NotImplementedError


        # compute logits
        criterion: bar_distribution.BarDistribution = self.model.criterion
        x_predict = torch.cat([x_given, x_eval], dim=0)


        logits_list = []
        for x_feed in torch.split(x_predict, max_dataset_size, dim=0):
            x_full_feed = torch.cat([x_given, x_feed], dim=0).unsqueeze(1)
            y_full_feed = y_given.unsqueeze(1)
            if ensemble_log_dims == '01':
                x_full_feed = log01_batch(x_full_feed)
            elif ensemble_log_dims == 'global01' or ensemble_log_dims is True:
                x_full_feed = log01_batch(x_full_feed, input_between_zero_and_one=True)
            elif ensemble_log_dims == '01-10':
                x_full_feed = torch.cat((log01_batch(x_full_feed)[:, :-1], log01_batch(1. - x_full_feed)), 1)
            elif ensemble_log_dims == 'norm':
                x_full_feed = lognormed_batch(x_full_feed, len(x_given))
            elif ensemble_log_dims is not False:
                raise NotImplementedError

            if ensemble_feature_rotation:
                x_full_feed = torch.cat([x_full_feed[:, :, (i+torch.arange(x_full_feed.shape[2])) % x_full_feed.shape[2]] for i in range(x_full_feed.shape[2])], dim=1)

            if ensemble_input_rank_transform == 'train' or ensemble_input_rank_transform is True:
                x_full_feed = torch.cat([rank_transform(x_given, x_full_feed[:,i,:])[:,None] for i in range(x_full_feed.shape[1])] + [x_full_feed], dim=1)

            if ensemble_power_transform:
                assert apply_power_transform is False
                y_full_feed = torch.cat((general_power_transform(y_full_feed, y_full_feed, power_transform_eps), y_full_feed), dim=1)


            if style is not None:
                if callable(style):
                    style = style()

                if isinstance(style, torch.Tensor):
                    style = style.to(x_full_feed.device)
                else:
                    style = torch.tensor(style, device=x_full_feed.device).view(1, 1).repeat(x_full_feed.shape[1], 1)


            logits = self.model(
                (style,
                x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]),
                y_full_feed.repeat(1,x_full_feed.shape[1])),
                single_eval_pos=len(x_given),
                layers_info = layers_info
            )
            if ensemble_type == 'mean_probs':
                logits = logits.softmax(-1).mean(1, keepdim=True).log_()  # (num given + num eval, 1, num buckets)

            logits_list.append(logits)  # (< max_dataset_size, 1 , num_buckets)
        logits = torch.cat(logits_list, dim=0) # (num given + num eval, 1 or (num_features+1), num buckets)
        del logits_list, x_full_feed
        if torch.isnan(logits).any():
            print('nan logits')
            #print(f"y_given: {y_given}, x_given: {x_given}, x_eval: {x_eval}")
            #print(f"logits: {logits}")
            return torch.zeros_like(x_eval[:,0])

        #logits = model((torch.cat([x_given, x_given, x_eval], dim=0).unsqueeze(1),
        #               torch.cat([y_given, torch.zeros(len(x_eval)+len(x_given), device=y_given.device)], dim=0).unsqueeze(1)),
        #               single_eval_pos=len(x_given))[:,0] # (N + M, num_buckets)
        logits_given = logits[:len(x_given)]
        logits_eval = logits[len(x_given):]
        self.target_logits = logits_eval
        #tau = criterion.mean(logits_given)[torch.argmax(y_given)] # predicted mean at the best y
        if predicted_mean_fbest:
            tau = criterion.mean(logits_given)[torch.argmax(y_given)].squeeze(0)
        else:
            tau = torch.max(y_given)
        #log_ei = torch.stack([criterion.ei(logits_eval[:,i], noisy_best_f[i]).log() for i in range(len(logits_eval))],0)

        def acq_ensembling(acq_values): # (points, ensemble dim)
            return acq_values.max(1).values

        if isinstance(acq_function, (dict,list)):
            acq_function = acq_function[style]

        if acq_function == 'ei':
            acq_value = acq_ensembling(criterion.ei(logits_eval, tau))
        elif acq_function == 'ei_or_rand':
            if torch.rand(1).item() < 0.5:
                acq_value = torch.rand(len(x_eval))
            else:
                acq_value = acq_ensembling(criterion.ei(logits_eval, tau))
        elif acq_function == 'pi':
            acq_value = acq_ensembling(criterion.pi(logits_eval, tau))
        elif acq_function == 'ucb':
            acq_function = criterion.ucb
            if ucb_rest_prob is not None:
                acq_function = lambda *args: criterion.ucb(*args, rest_prob=ucb_rest_prob)
            acq_value = acq_ensembling(acq_function(logits_eval, tau))
        elif acq_function == 'mean':
            acq_value = acq_ensembling(criterion.mean(logits_eval))
        elif acq_function.startswith('hebo'):
            noise, upsi, delta, eps = (float(v) for v in acq_function.split('_')[1:])
            noise = y_given_std * math.sqrt(2 * noise)
            kappa = math.sqrt(upsi * 2 * ((2.0 + x_given.shape[1] / 2.0) * math.log(max(1, len(x_given))) + math.log(
                3 * math.pi ** 2 / (3 * delta))))
            rest_prob = 1. - .5 * (1 + torch.erf(torch.tensor(kappa / math.sqrt(2), device=logits.device)))
            ucb = acq_ensembling(criterion.ucb(logits_eval, None, rest_prob=rest_prob)) \
                + torch.randn(len(logits_eval), device=logits_eval.device) * noise
            noisy_best_f = tau + eps + \
                        noise * torch.randn(len(logits_eval), device=logits_eval.device)[:, None].repeat(1, logits_eval.shape[1])

            log_pi = acq_ensembling(criterion.pi(logits_eval, noisy_best_f).log())
            # log_ei = torch.stack([criterion.ei(logits_eval[:,i], noisy_best_f[i]).log() for i in range(len(logits_eval))],0)
            log_ei = acq_ensembling(criterion.ei(logits_eval, noisy_best_f).log())

            acq_values = torch.stack([ucb, log_ei, log_pi], dim=1)

            def is_pareto_efficient(costs):
                """
                Find the pareto-efficient points
                :param costs: An (n_points, n_costs) array
                :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
                """
                is_efficient = torch.ones(costs.shape[0], dtype=bool, device=costs.device)
                for i, c in enumerate(costs):
                    if is_efficient[i]:
                        is_efficient[is_efficient.clone()] = (costs[is_efficient] < c).any(
                            1)  # Keep any point with a lower cost
                        is_efficient[i] = True  # And keep self
                return is_efficient

            acq_value = is_pareto_efficient(-acq_values)
        else:
            raise ValueError(f'Unknown acquisition function: {acq_function}')

        max_acq = acq_value.max()

        return acq_value if return_actual_ei else (acq_value == max_acq)