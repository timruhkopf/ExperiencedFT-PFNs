import contextlib

from pfns4bo import transformer
from pfns4bo  import bar_distribution
import torch
import scipy
import math
from sklearn.preprocessing import power_transform, PowerTransformer
import numpy as np


def log01(x, eps=.0000001, input_between_zero_and_one=False):
    logx = torch.log(x + eps)
    if input_between_zero_and_one:
        return (logx - math.log(eps)) / (math.log(1 + eps) - math.log(eps))
    return (logx - logx.min(0)[0]) / (logx.max(0)[0] - logx.min(0)[0])

def log01_batch(x, eps=.0000001, input_between_zero_and_one=False):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        x[:, b, b] = log01(x[:, b, b], eps=eps, input_between_zero_and_one=input_between_zero_and_one)
    return x

def lognormed_batch(x, eval_pos, eps=.0000001):
    x = x.repeat(1, x.shape[-1] + 1, 1)
    for b in range(x.shape[-1]):
        logx = torch.log(x[:, b, b]+eps)
        x[:, b, b] = (logx - logx[:eval_pos].mean(0))/logx[:eval_pos].std(0)
    return x

def _rank_transform(x_train, x):
    assert len(x_train.shape) == len(x.shape) == 1
    relative_to = torch.cat((torch.zeros_like(x_train[:1]),x_train.unique(sorted=True,), torch.ones_like(x_train[-1:])),-1)
    higher_comparison = (relative_to < x[...,None]).sum(-1).clamp(min=1)
    pos_inside_interval = (x - relative_to[higher_comparison-1])/(relative_to[higher_comparison] - relative_to[higher_comparison-1])
    x_transformed = higher_comparison - 1 + pos_inside_interval
    return x_transformed/(len(relative_to)-1.)

def rank_transform(x_train, x):
    assert x.shape[1] == x_train.shape[1], f"{x.shape=} and {x_train.shape=}"
    # make sure everything is between 0 and 1
    assert (x_train >= 0.).all() and (x_train <= 1.).all(), f"{x_train=}"
    assert (x >= 0.).all() and (x <= 1.).all(), f"{x=}"
    return_x = x.clone()
    for feature_dim in range(x.shape[1]):
        return_x[:, feature_dim] = _rank_transform(x_train[:, feature_dim], x[:, feature_dim])
    return return_x



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


def general_transforms(
    x_given, y_given, x_eval,
    apply_power_transform=True,
    znormalize=False,
    pre_normalize=False,
    pre_znormalize=False,
    input_znormalize=False,
    max_dataset_size=10_000,
    remove_features_with_one_value_only=False,
    ensemble_type='mean_probs',
    input_power_transform=False,
    power_transform_eps=.0,
    input_power_transform_eps=.0,
    outlier_stretching_interval=0.0,
    verbose=False,
    unsafe_power_transform=False,
    input_rank_transform=False
):

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
    return x_given, y_given, x_eval


def general_PFN_extrapolation(model: transformer.TransformerModel, x_given, y_given, x_eval, style = None,  ensemble_log_dims=False, max_dataset_size=10_000,
    ensemble_type='mean_probs', # in ('mean_probs', 'max_acq')
    ensemble_power_transform=False, ensemble_feature_rotation=False,ensemble_input_rank_transform=False,):
    assert ensemble_type in ('mean_probs', 'max_acq')
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

        logits = model(
            (style,
                x_full_feed.repeat_interleave(dim=1, repeats=y_full_feed.shape[1]),
                y_full_feed.repeat(1,x_full_feed.shape[1])),
            single_eval_pos=len(x_given)
        )
        if ensemble_type == 'mean_probs':
            logits = logits.softmax(-1).mean(1, keepdim=True).log_()  # (num given + num eval, 1, num buckets)
        logits_list.append(logits)  # (< max_dataset_size, 1 , num_buckets)
    logits = torch.cat(logits_list, dim=0) # (num given + num eval, 1 or (num_features+1), num buckets)
    del logits_list, x_full_feed
    if torch.isnan(logits).any():
        print('nan logits')
        print(f"y_given: {y_given}, x_given: {x_given}, x_eval: {x_eval}")
        print(f"logits: {logits}")
        return torch.zeros_like(x_eval[:,0])

    logits_given = logits[:len(x_given)]
    logits_eval = logits[len(x_given):]

    return logits_given, logits_eval

def only_acq_function(model,logits_given, logits_eval, y_given, predicted_mean_fbest=False, return_actual_ei=False, acq_function='ei', ucb_rest_prob=.05, **kwargs):
    criterion: bar_distribution.BarDistribution = model.criterion
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


def our_acq_function(model,logits_given, logits_eval, y_given, list_meta_tasks, predicted_mean_fbest=False, return_actual_ei=False, **kwargs):
    criterion: bar_distribution.BarDistribution = model.criterion
    if predicted_mean_fbest:
        tau = criterion.mean(logits_given)[torch.argmax(y_given)].squeeze(0)
    else:
        tau = torch.max(y_given)
    

    def acq_ensembling(acq_values): # (points, ensemble dim)
        return acq_values.max(1).values

    acq_value = acq_ensembling(criterion.ei(logits_eval, tau))

    max_acq = acq_value.max()
    return acq_value if return_actual_ei else (acq_value == max_acq)



# class EnsembleTransformerBOMethod:
#     def __init__(self, model, meta_data, acq_f=general_acq_function, device='cpu:0', fit_encoder = None, **kwargs):
#         self.model = model
#         self.device = device
#         self.kwargs = kwargs
#         self.acq_function = acq_f
#         self.normalizer = general_transforms
#         self.predictor = general_PFN_extrapolation
#         self.acq_function = our_acq_function
#         self.fit_encoder = fit_encoder
#         self.meta_X = to_tensor(np.stack([ item["X"]   for k, item in meta_data.items()], axis=0)).to(torch.float32)
#         self.meta_y =  to_tensor(np.stack([ item["y"]   for k, item in meta_data.items()], axis=0)).to(torch.float32)  
#         print(f"EnsembleTransformerBOMethod: meta_X shape: {self.meta_X.shape}, meta_y shape: {self.meta_y.shape}")

#     @torch.no_grad()
#     def observe_and_suggest(self, X_obs, y_obs, X_pen, return_actual_ei=False):
#         # assert X_pen is not None
#         # assumptions about X_obs and X_pen:
#         # X_obs is a numpy array of shape (n_samples, n_features)
#         # y_obs is a numpy array of shape (n_samples,), between 0 and 1
#         # X_pen is a numpy array of shape (n_samples_left, n_features)
#         X_obs = to_tensor(X_obs, device=self.device).to(torch.float32)
#         y_obs = to_tensor(y_obs, device=self.device).to(torch.float32).view(-1)
#         X_pen = to_tensor(X_pen, device=self.device).to(torch.float32)

#         print(f"X_obs shape: {X_obs.shape}, y_obs shape: {y_obs.shape}, X_pen shape: {X_pen.shape}")

#         assert len(X_obs) == len(y_obs), "make sure both X_obs and y_obs have the same length."

#         self.model.to(self.device)

#         if self.fit_encoder is not None:
#             w = self.fit_encoder(self.model, X_obs, y_obs)
#             X_obs = w(X_obs)
#             X_pen = w(X_pen)

#         with (torch.cuda.amp.autocast() if self.device[:3] != 'cpu' else contextlib.nullcontext()):
#             list_meta_tasks = []
#             for i in range(len(self.meta_X)):
#                 x_given, y_given, x_eval = self.normalizer(self.meta_X[i], self.meta_y[i], X_pen, **self.kwargs)
#                 logits_given, logits_eval = self.predictor(self.model,  x_given, y_given, x_eval, **self.kwargs)
#                 list_meta_tasks.append((logits_given, logits_eval, y_given))

#             x_given, y_given, x_eval = self.normalizer(X_obs, y_obs, X_pen, **self.kwargs)
#             logits_given, logits_eval = self.predictor(self.model,  x_given, y_given, x_eval, **self.kwargs)    
#             acq_values = self.acq_function(self.model,  logits_given, logits_eval, y_given, list_meta_tasks, **self.kwargs).cpu().clone()  # bool array
#             acq_mask = acq_values.max() == acq_values

#         possible_next = torch.arange(len(X_pen))[acq_mask]
#         if len(possible_next) == 0:
#             possible_next = torch.arange(len(X_pen))

#         r = possible_next[torch.randperm(len(possible_next))[0]].cpu().item()


#         if return_actual_ei:
#             return r, acq_values
#         else:
#             return r