from functools import partial

import torch

from model.components.imputor import sample_logits
from model.probability_conv.convolver import DistributionConvolver
from model.strategies.abstract_strategy import AbstractStrategy


class ProbabilisticMetaContextStrategy(AbstractStrategy):
    """
    We can try to model the covariance between tasks using a PFN, where we
    take the marginal models for each related (and target task) and sample data from them
    at  grid locations. Collecting repeated samples from these marginals
    """
    __name__ = "ProbabilisticMetaContextStrategy"

    def __init__(self, imputer, model_avg='mean', n_samples=10, add_hp=True, support='sobol',
                 **kwargs):
        """

        :param imputer:
        :param model_avg:
        :param n_samples: The number batch parallel processed marginal samples per location --
        basically monte carlo samples from the marginal ppds that will generate surfaces to reflect
        the uncertainty contained in the PPDs. Note however, that marginal sampling ignore
        covariance.
        :param add_hp: bool, whether to add the hp dimensions to the landmarking features from
        the related tasks.
        :param support: str, either 'sobol' or 'lhd', determines the support sampling strategy for the support set
         locations. These will help determine the transfer from related tasks to the target task.
        :param kwargs: n_support_samples: if provided, will use this many support samples,
         otherwise will use 10 * hpD.
         condensed: bool, whether to use the condensed variant, where after the first pass
         of sampling from the related tasks ppds and using those exclusively for prediciton of
         the target task's y, We take those predictions as a condensate of all related tasks
         and use them again as a single  landmarking dim in conjunction with the hp locations for a
         second pass.
        """
        self.imputer = imputer
        self.model_avg = model_avg
        self.n_samples = n_samples
        self.add_hp = add_hp

        self.support = support

        # # will dynamically be computed as 10 * hpD if None
        self.n_support_samples = kwargs.get('n_support_samples', None)

        assert support in ['sobol', 'lhd'], "Support must be either 'sobol' or 'lhd'"

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        super().__post_init__(
            parent_model=parent_model,
            model=model,
            related_context=related_context,
            callbacks=callbacks,
            logger=logger,
            device=device,
            **kwargs
        )

        self.imputer.__post_init__(
            parent_model=self,
            model=self.model,
            criterion=self.model.criterion,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device
        )

    def __call__(self, x_train, x_test, y_train, inc, **kwargs):
        """
        Probabilistic Joint Context Model with Prior Transfer

        We want the following parings

        Context --------------------------
        [target | context], * indicates target task, r indicates related task
        hat implies an imputation

        1. [y^*(x_train)| \hat{y}_r(x_train), x_train]  -> target train features
        This provides the grounds for the target task in terms of training data
        as well as comparing its observed performance against the related tasks' imputed performance
        as a landmark hyperparameter feature.

        2. [\hat{y}^*(x_r) | y_r(x_r),  x_r] -> related train features
        This provides the grounds for the generalization from related task to the target task; i.e.
        this way we can leap from what the related tasks observations to what we believe will
        happen under the target

        Query: ------------------------
        3. [ ? | \hat{y}_r(x_test), x_test ] -> target test features

        Notes:
            y^*(x_train) = y_train
            \hat{y}_r(x_train) = imputed y values for related tasks at target train locations
            \hat{y}^*(x_r) = imputed y values for target task at related tasks' locations
            y_r(x_r) = related tasks' observed y values at their own locations
            \hat{y}_r(x_test) = imputed y values for related tasks at target test locations

        Obviously, any imputation done here creates a "reality" / scenario that has no
        probabilistic perspective or knowledge of the implied uncertainty that generated it.
        I.e. the imputed observations under the target are assumed to be true values from the
        perspective of the model.
        By Monte-Carlo sampling, (especially for the target task imputations at related tasks' locations)
        we try to mitigate this effect by providing multiple scenarios and averaging over the
        resulting PPD logits. This way we reflect the uncertainty contained in the PPD of the
        target model.

        # IMPORTANT IMPLEMENTATION NOTE:
         The FT-PFN model expects [idx, fidelity, hp1, hp2, ...] as input features.
         So we need to make sure that any landmarking feature is appended as a
         regular hp dimension at the end of the feature vector. Otherwise the
         encoder will mess with us.
        """

        # 2. get sampled functions under target task at related_tasks' locations
        # p( y | Q=x_related, \tau^* ) -->  \hat{y}^*(x_r)
        _, y_target_x_related = self.imputer(
            x_train=x_train.repeat(1, self.num_related, 1),
            x_test=self.related_context.x,
            y_train=y_train.repeat(1, self.num_related),
            n_samples=self.n_samples,
        )  # (T_related, num_related, n_samples)

        # 1 & 3. get sampled functions under the prior at target task's locations
        # p( y | Q=x_test, \tau_i ) -->  \hat{y}_r(x_test)
        # p( y | Q=x_train, \tau_i ) -->  \hat{y}_r(x_train)
        _, y_r = self.imputer(
            x_train=self.related_context.x,
            x_test=torch.cat([
                x_train.repeat(1, self.num_related, 1),  # 2
                x_test.repeat(1, self.num_related, 1)  # 3
            ], dim=0),
            y_train=self.related_context.y,
            n_samples=self.n_samples,
        )  # (T_train + T_test, num_related, n_samples)




        # split according to x_test:
        n_support_samples = x_train.shape[0]
        y_r_x_train = y_r[:n_support_samples, :]  # 2 # (T_train, num_related, n_samples)
        y_r_x_test = y_r[n_support_samples:, :]  # 3 # (T_test, num_related, n_samples)

        # (4) likelihood of target data under related task
        yasdf =  self.get_prior_model(x_train.repeat(1, self.num_related, 1))
        related_nll= [self.model.criterion(yasdf[:, b], y_train).mean() for b in range(
                self.num_related)]


        # if self.add_hp:
        #     hp_dim = None
        #
        # else:
        #     hp_dim = 2  # id + fidelity dim in ifbo

        # context dim (hp) is appended with landmark features from related tasks
        # Here we only ever add one task at a time in D, but process all of them in batch parallel
        # (Training features)  -----------------------------
        # target: y* based on (\hat{y}_r(x*), x*)
        # FIXME: REMOVE TASK FLAG
        TASK = 1

        train_features = torch.cat([

            x_train[..., :].repeat(1, self.n_samples, 1),  # (T, n_samples, hp)
            # .repeat(1, self.n_samples * self.num_related, 1), id + fidelity dim
            y_r_x_train[:, :TASK].permute(0, 2, 1),  # (T, num_samples, 1)
            # .permute(0, 2, 1).reshape(-1, self.n_samples * self.num_related, 1),
        ], dim=-1)  # (T, n_samples, hp + 1)

        # 2. target: \hat{y}*, based on (y_r(x_r) , x_r) ,
        related_features = torch.cat([
            self.related_context.x[:, :TASK],  # (T_related, 1, hp)
            self.related_context.y[:, :TASK].unsqueeze(-1),  # (T_related, 1, 1)
        ], dim=-1).repeat(1, self.n_samples, 1)  # (T, n_samples , hp + 1)

        # (Targets) -----------------------------
        # ( y*, \hat{y}^*(x_r) )
        targets = torch.cat([
            y_train.unsqueeze(-1).repeat(1, self.n_samples, 1),  # (T, n_samples, 1)
            # .repeat(1, self.n_samples * self.num_related,1),
            y_target_x_related[:, :TASK].permute(0, 2, 1),  # (T_related, n_samples, 1)
            # .permute(0, 2, 1).reshape(-1, self.n_samples * self.num_related, 1)
        ], dim=0).squeeze(-1)  # (T + T_related, n_samples)

        # (Query features) -----------------------------
        # (\hat{y}_r(x_test), x_test)
        query_features = torch.cat([
            x_test.repeat(1, self.n_samples, 1) , # (T_test, n_samples, hp)
            # .repeat(1, self.n_samples * self.num_related, 1),  # id + fidelity dim
            y_r_x_test[:, :TASK].permute(0, 2, 1),  # (T_test, n_samples, 1)
            # .permute(0, 2, 1).reshape(-1, self.n_samples * self.num_related, 1),
        ], dim=-1)  # (T_test, n_samples, hp + 1)

        # p( y | Q=x_test, \tau^*, \tau_i ) after reshaping correctly and then averaging over
        # samples per related task!
        logits = self.model(
            (
                torch.cat([train_features, related_features, query_features], dim=0),
                targets
            ),
            single_eval_pos=train_features.shape[0] + related_features.shape[0],
        )

        logits = logits.mean(dim=1) #[:,0] #

        #
        # # undo the batch parallel trick and average per task over samples:
        # logits = logits.reshape(
        #     x_test.shape[0], self.n_samples, self.num_related, logits.shape[-1]
        # ).permute(0,2,1, 3).mean(dim=2)  # (T_test, num_related, D_logits)

        # Only Manual call to plot here for debugging, otherwise recursion!
        #     self.plot(x_train, x_test, y_train, inc)

        return logits

    def plot(self, x_train, x_test, y_train, inc):
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import matplotlib.pyplot as plt
        from plotly.offline import plot
        from callbacks.callback_1d_projection import get_lower_upper_surfaces

        grid_size = 50
        x1 = np.linspace(0, 1, grid_size)
        x2 = np.linspace(0, 1, grid_size)
        X1, X2 = np.meshgrid(x1, x2)
        config_id = torch.tensor((X1 * 999).ravel()).floor()
        X_grid = torch.vstack([config_id, torch.tensor(X1.ravel()), torch.tensor(X2.ravel())]).T
        X_grid = X_grid.to(self.device).float().unsqueeze(1)

        fig = make_subplots(
            rows=1, cols=self.num_related + 2,
            subplot_titles=["Target Model", "Merged Model"] + [f"Prior Model {i + 1}" for i in
                                                               range(self.num_related)],
            #                    "Prior Points"] * (self.num_related + 1),
            specs=[[{"type": "surface"}] * (self.num_related + 2)],
        )

        # (surfaces) ----------------------------------
        surf_prior_logits = self.get_prior_model(x_test=X_grid.repeat(1, self.num_related, 1))
        surf_target_logits = self.get_target_model(x_train=x_train, y_train=y_train, x_test=X_grid)
        surf_joint_logits = self.__call__(x_train=x_train, x_test=X_grid, y_train=y_train, inc=None)

        # FIXME: remove both flags
        TASK =0
        SAMPLE = 1
        # surf_joint_logits = surf_joint_logits[:, SAMPLE, TASK, ...]
        # surf_joint_logits = surf_joint_logits[:, SAMPLE, ...]
        crit = self.model.criterion
        get_lower_upper_surfaces(
            criterion=crit, logits=surf_target_logits,
            row=1, col=1, fig=fig, name='Target Model',
            x1=X1, x2=X2, grid_size=grid_size
        )
        get_lower_upper_surfaces(
            criterion=crit, logits=surf_joint_logits,
            row=1, col=2, fig=fig, name='Merged Model',
            x1=X1, x2=X2, grid_size=grid_size
        )
        for i in range(self.num_related):
            get_lower_upper_surfaces(
                criterion=crit, logits=surf_prior_logits[:, i:i + 1, :],
                row=1, col=i + 3, fig=fig, name=f'Prior Model {i + 1}',
                x1=X1, x2=X2, grid_size=grid_size
            )

        # (prior scatter points) ----------------------------------
        for i in range(self.num_related):
            fig.add_trace(go.Scatter3d(
                x=self.related_context.x[:, i, 1].cpu().numpy(),
                y=self.related_context.x[:, i, 2].cpu().numpy(),
                z=self.related_context.y[:, i].cpu().numpy(),
                mode='markers',
                marker=dict(size=2, color='red'),
                name=f'Prior Points {i + 1}'
            ), row=1, col=i + 3)

        # (target scatter points) ----------------------------------
        # coloured based
        for col in [1, 2]:
            fig.add_trace(go.Scatter3d(
                x=x_train[:, 0, 1].cpu().numpy(),
                y=x_train[:, 0, 2].cpu().numpy(),
                z=y_train.squeeze().cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=2,
                    color=torch.arange(0, x_train.shape[0]).cpu().numpy(),
                    colorscale='Plasma',  # Or any colorscale you like
                    colorbar=dict(title='Step')
                ),
                name='Target Points'
            ), row=1, col=col)

        # (x_test points to point in target) ----------------------------------
        target_test_logits = self.get_target_model(x_train=x_train, y_train=y_train, x_test=x_test)
        target_test_pi = self.model.criterion.pi(
            target_test_logits.squeeze(),
            best_f=inc,
            maximize=True
        ).cpu().numpy()

        fig.add_trace(go.Scatter3d(
            x=x_test[:, 0, 1].cpu().numpy(),
            y=x_test[:, 0, 2].cpu().numpy(),
            z=self.model.criterion.median(target_test_logits).squeeze(),
            mode='markers',
            marker=dict(size=2, color=target_test_pi, colorscale='Viridis', colorbar=dict(
                title='PI')),
            name='Test Points PI'
        ), row=1, col=1)

        # (x_test points to point in merged model) ----------------------------------
        merged_test_logits = self.__call__(
            x_train=x_train, x_test=x_test, y_train=y_train,
            inc=None
        )
        # FIXME: REMOVE ME
        # merged_test_logits = merged_test_logits[:, SAMPLE, TASK, ...]
        # merged_test_logits = merged_test_logits[:, SAMPLE, ...]
        merged_test_pi = self.model.criterion.pi(
            merged_test_logits,
            best_f=inc,
            maximize=True
        ).cpu().numpy()

        fig.add_trace(go.Scatter3d(
            x=x_test[:, 0, 1].cpu().numpy(),
            y=x_test[:, 0, 2].cpu().numpy(),
            z=self.model.criterion.median(merged_test_logits),
            mode='markers',
            marker=dict(size=2, color=merged_test_pi, colorscale='Viridis', colorbar=dict(
                title='PI')),
            name='Test Points PI'
        ), row=1, col=2)

        # (interactive plotting with sample data) ------------------
        # y_r_x_train samples
        fig.add_trace(go.Scatter3d(
            x=x_train[:, 0, 1].cpu().numpy(),
            y=x_train[:, 0, 2].cpu().numpy(),
            z=y_r_x_train[:, 0, SAMPLE].cpu().numpy(),
            mode='markers',
            marker=dict(size=2, color='cyan'),

        ), row=1, col=2)

        # y_target_x_related samples
        fig.add_trace(go.Scatter3d(
            x=self.related_context.x[:, 0, 1].cpu().numpy(),
            y=self.related_context.x[:, 0, 2].cpu().numpy(),
            z=y_target_x_related[:, 0, SAMPLE].cpu().numpy(),
            mode='markers',
            marker=dict(size=2, color='orange'),

        ), row=1, col=2)

        # y_r_x_test samples
        fig.add_trace(go.Scatter3d(
            x=x_test[:, 0, 1].cpu().numpy(),
            y=x_test[:, 0, 2].cpu().numpy(),
            z=y_r_x_test[:, 0, SAMPLE].cpu().numpy(),
            mode='markers',
            marker=dict(size=2, color='pink'),

        ), row=1, col=2)

        # train = torch.cat([train_features, related_features], dim=0)
        #
        #
        # # these two should be aligned in T dim. plot the data points
        # # notice, that train is (T + T_related, n_samples, hp + 1)
        # # let us first plot
        # fig.add_trace(go.Scatter3d(
        #     x=train[:, 0, 1].cpu().numpy(),
        #     y=train[:, 0, 2].cpu().numpy(),
        #     z=targets[:, 0].cpu().numpy(),
        #     mode='markers',
        #     marker=dict(size=3, color='black'),
        #     name='Meta-Context Training Points'
        # ), row=1, col=2
        # )

        # (layout) ----------------------------------
        fig.update_layout(
            height=700, width=2400, title_text=f"Step {x_train.shape[0]} Meta-Context Models",
            scene1=dict(
                xaxis_title='fidelity',
                yaxis_title='lambda',
                zaxis_title='f(x,lambda)',
            ),
            scene2=dict(
                xaxis_title='fidelity',
                yaxis_title='lambda',
                zaxis_title='f(x,lambda)',
            )
        )
        # fig.show()
        plot(fig)
        plt.clf()


class ProbabilisticMetaContextStrategyV2(ProbabilisticMetaContextStrategy):
    def __init__(self, *args, **kwargs):
        """
        Here we basically run the ProbabilisticMetaContextStrategy with add_hp=False initially,
        which will give us the regression y_target ~ f({y_related_i}_i). Taking the the resulting PPD
        as a condensate of all the related tasks, we can then use it again as a single landmarking feature
        in a second pass y_target ~ f(hp, y_condensate), where we can now also add the hp dimensions again.

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.condensed = True
        if self.condensed:
            self.add_hp = False

    def __call__(self, x_train, x_test, y_train, inc, **kwargs):
        T, B, hpD = x_train.shape

        # 1. get samples from the marginals of related tasks at target task's locations
        y_train_prior = self.imputer(
            x_train=self.related_context.x,
            x_test=x_train.repeat(1, self.num_related, 1),
            y_train=self.related_context.y,
            n_samples=self.n_samples,
        )

        # 2. collect a random support set
        if self.n_support_samples is None:
            self.n_support_samples = 10 * hpD  # heuristic

        if self.support == 'sobol':
            sobol_engine = qmc.Sobol(d=hpD, scramble=True)
            sobol_sample = sobol_engine.random(n=self.n_support_samples)

        elif self.support == 'lhd':
            lhs_engine = qmc.LatinHypercube(d=hpD)
            sobol_sample = lhs_engine.random(n=self.n_support_samples)

        support_x = torch.tensor(sobol_sample, device=self.device).float().unsqueeze(1)
        support_x[..., 0] = 999.  # hp id dimension in IFBO!

        # 3. get imputed y values for related tasks at support set locations
        y_prior = self.imputer(
            x_train=self.related_context.x,
            x_test=torch.cat([
                support_x.repeat(1, self.num_related, 1),
                x_test.repeat(1, self.num_related, 1)
            ], dim=0),

            y_train=self.related_context.y,
            n_samples=self.n_samples,
        )

        y_support_prior = y_prior[:self.n_support_samples, :]  # (n, num_related)
        y_test_prior = y_prior[self.n_support_samples:, :]  # (T', num_related)

        # 4. imputed y values for target task at support set locations
        y_support_target = self.imputer(
            x_train=x_train,
            x_test=support_x,
            y_train=y_train,
            n_samples=self.n_samples
        )

        if self.n_samples == 1:
            y_train_prior = y_train_prior.unsqueeze(-1)  # (T, 1, num_related)
            y_support_target = y_support_target.unsqueeze(-1)  #
            y_support_prior = y_support_prior.unsqueeze(-1)  # (n, 1, num_related)
            y_test_prior = y_test_prior.unsqueeze(-1)  # (T', 1

        y_train = y_train.unsqueeze(1)  # (T, 1)

        # samples are batch dim from now one
        y_support_target = y_support_target.permute(0, 2, 1)
        y_support_prior = y_support_prior.permute(0, 2, 1)
        y_train_prior = y_train_prior.permute(0, 2, 1)
        y_test_prior = y_test_prior.permute(0, 2, 1)

        if self.add_hp:
            hp_dim = None

        else:
            hp_dim = 2  # id + fidelity dim in ifbo

        # 5. collect all the imputed y values and use them as landmarking feature vectors.
        support_features = torch.cat([
            support_x[..., :hp_dim].repeat(1, self.n_samples, 1),  # id + fidelity dim
            y_support_prior,
        ], dim=-1)  # (n, 1 + 1 + num_related)

        features_train = torch.cat([
            x_train[..., :hp_dim].repeat(1, self.n_samples, 1),  # id + fidelity dim
            y_train_prior,
        ], dim=-1)

        final_train_features = torch.cat([features_train, support_features], dim=0)
        final_train_targets = torch.cat([y_train.repeat(1, self.n_samples, 1), y_support_target],
                                        dim=0)

        test_features = torch.cat([
            x_test[..., :hp_dim].repeat(1, self.n_samples, 1),  # id + fidelity dim
            y_test_prior,  # (T', 1 + num_related)

        ], dim=-1)
        test, train, support = test_features.shape[0], features_train.shape[0], \
            support_features.shape[0]
        test_features = torch.cat([test_features, features_train, support_features], dim=0)

        # use batch parallel sampled data from the related and target tasks
        # and average the resulting sample based logits
        logits = self.model(
            (
                torch.cat([final_train_features, test_features], dim=0),
                final_train_targets.squeeze(-1)
            ),
            single_eval_pos=final_train_features.shape[0],
        )

        samples = torch.stack([
            sample_logits(logits[:, b, :], n_samples=10, borders=self.model.criterion.borders)
            for b in range(logits.shape[1])
        ], dim=1)

        b = self.n_samples * 10

        test_samples = samples[:test, :].reshape(-1, b, 1)
        train_samples = samples[test:test + train, :].reshape(-1, b, 1)
        support_samples = samples[test + train:, :].reshape(-1, b, 1)

        train_context = torch.cat([x_train.repeat(1, b, 1), train_samples], dim=-1)
        support_context = torch.cat([support_x.repeat(1, b, 1), support_samples], dim=-1)
        test_context = torch.cat([x_test.repeat(1, b, 1), test_samples], dim=-1)

        final_train_features = torch.cat([train_context, support_context], dim=0)
        final_test_features = test_context
        final_train_targets = torch.cat([y_train.repeat(1, b, 1), support_samples], dim=0)

        logits = self.model(
            (
                torch.cat([final_train_features, final_test_features], dim=0),
                final_train_targets.squeeze(-1)
            ),
            single_eval_pos=final_train_features.shape[0],
        ).mean(dim=1)

        return logits


class SplitMetaContextStrategy(AbstractStrategy):
    __name__ = "SplitMetaContextStrategy"

    def __init__(self, imputer, model_avg='mean'):
        self.imputer = imputer
        self.model_avg = model_avg

    def __post_init__(self, parent_model, model, related_context, callbacks, logger, device,
                      **kwargs):
        super().__post_init__(
            parent_model=parent_model,
            model=model,
            related_context=related_context,
            callbacks=callbacks,
            logger=logger,
            device=device,
            **kwargs
        )

        self.imputer.__post_init__(
            parent_model=self,
            model=self.model,
            criterion=self.model.criterion,
            related_context=self.related_context,
            logger=self.logger,
            device=self.device
        )

    def get_prior_augmented_target_model(self, x_train, x_test, y_train):
        # how does x_test look under the related tasks?
        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y)

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')

        imputed_y = self.parent_model.interim_results['imputed_y'].to(self.device)

        # Here we concatenate one imputed prior value to the HP dimension, but repeat it for all related tasks.
        features = torch.concat(
            [x_train.repeat(1, self.num_related, 1), imputed_y.unsqueeze(-1)], dim=-1
        )
        features_test = torch.concat(
            [x_test.repeat(1, self.num_related, 1), imputed_y_test.unsqueeze(-1)], dim=-1
        )
        return self.model(
            (
                torch.cat([features, features_test]),
                y_train.repeat(1, self.num_related)
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )

    def __call__(self, x_train, x_test, y_train, inc):
        """

        This strategy uses imputed y values under the prior and concatenates them to the
        x_train and x_test, but it only ever uses a single related task at a time.
        We abuse the batch dim to do it concurrently for all related tasks.

        """

        step = x_train.shape[0]

        target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
        target_logits = target_model(x_test=x_test)

        prior_augmented_target_model = partial(
            self.get_prior_augmented_target_model,
            x_train=x_train, y_train=y_train
        )
        prior_logits = prior_augmented_target_model(x_test=x_test)

        predictions = torch.cat((target_logits, prior_logits), dim=1)

        self.parent_model.interim_results.update({
            'prior_augmented_target_model': prior_augmented_target_model,
            'target_model': target_model,
            'past_x_test': x_test.cpu(),
            'logits': predictions.cpu(),
            # 'error_logits': target_logits,  # for plotting purposes
        })

        # TODO find a weighing!
        weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)

        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item()
                for i, w in enumerate(weights)}},
        )

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return (predictions * weights.unsqueeze(1)).sum(dim=1)


class JointMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointMetaContextStrategy"

    def __call__(self, x_train, x_test, y_train, inc):
        """
        Main idea behind this method is that we can use all of the related task y values (imputed
        for the x_test values of course) and concatenate them to the hp dimension.
        ideally, the model will learn to recognize whether the prior y dims are relevant.
        They are basically a probing feature that may or may not be relevant for the target task.


        # FIXME: DISADVANTAGE: WE CAN ONLY USE 10 DIMENSIONS IN TOTAL FOR THE INPUT FEATURES.
        # GIVEN THAT THE HP ALSO OCCUPY THIS SPACE, WE CAN ONLY USE 10 - HP_DIMENSIONS
        # FOR THE RELATED TASKS.

        :param x_train:
        :param x_test:
        :param y_train:
        :param inc:
        :return:
        """
        step, _, hp = x_train.shape

        # sample without the available priors based on their past surprise:

        avail_space = 10 - hp
        assert avail_space > 0, \
            (f'PFN input restriction violated: n_hp:{x_train.shape[-1]}. There are'
             '{self.num_related} related tasks, but they don\'t fit into the hp dim')
        if avail_space < self.num_related:
            weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)
            prior_weights = weights[1:]

            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )

            # now sample:
            prior_idx = torch.multinomial(
                prior_weights, num_samples=min(avail_space, self.num_related),
                replacement=False
            )
        else:
            prior_idx = torch.arange(self.num_related, device=self.device)

        imputed_y = self.parent_model.interim_results['imputed_y'].to(self.device)

        imputed_y_test = self.imputer(
            x_train=self.related_context.x[:, prior_idx, :],
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y[:, prior_idx])

        assert x_train.shape[-1] + imputed_y_test.shape[-1] <= 10, \
            (f'PFN input restriction violated: {x_train.shape[-1]}:x_train + '
             f'{imputed_y_test.shape[-1]}:imputed_y_test > 10.\n'
             'Recommendation: reduce the number of priors!')

        # here all the priors are directly concatenated to the HP dimensions!
        features = torch.concat([x_train, imputed_y.unsqueeze(1)], dim=-1)
        features_test = torch.concat([x_test, imputed_y_test.unsqueeze(1)], dim=-1)
        target_logits = self.model(
            (
                torch.cat([features, features_test]),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )

        self.parent_model.interim_results['logits'] = target_logits.cpu()  # for plotting purposes

        # TODO past surprise weighing by prior?

        for callback in self.callbacks:
            callback.on_final_weights(target_logits, weights)
        return target_logits


class JointNoHPMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointNoHPMetaContextStrategy"

    def __init__(self, imputer, prior_only=False, **kwargs):
        super().__init__(imputer=imputer, **kwargs)
        self.prior_only = prior_only

    def __call__(self, x_train, x_test, y_train, inc):
        """
        Here, we only will use the imputed prior y values as input to the model.
        We can augment the model prediction with that of a meta-unaware target model,
        solely based on the hp values.
        """

        step, _, hp = x_train.shape

        # sample without the available priors based on their past surprise:

        avail_space = 10
        assert avail_space > 0, \
            (f'PFN input restriction violated: n_hp:{x_train.shape[-1]}. There are'
             '{self.num_related} related tasks, but they don\'t fit into the hp dim')
        if avail_space < self.num_related:
            weights = self.parent_model.weights(x_train, x_test, y_train, inc, recompute=True)
            prior_weights = weights[1:]

            self.logger.log(
                {'metrics': 'weights', 'step': step,
                 **{f'weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )

            # now sample:
            # TODO instead of subsampling just batch the model calls over different contexts--
            # and then measure their respective surprises
            prior_idx = torch.multinomial(
                prior_weights, num_samples=min(avail_space, self.num_related),
                replacement=False
            )
        else:
            prior_idx = torch.aragne(self.num_related, device=self.device)
            weights = torch.ones(self.num_related + 1, device=self.device) / (self.num_related + 1)

        # Collect the y (train/test) values under the priors as "landmarking" features:
        imputed_y = self.parent_model.interim_results['imputed_y'].to(self.device)
        imputed_y = imputed_y[:, prior_idx]
        imputed_y = imputed_y.reshape(imputed_y.shape[0], 1 - 1)

        imputed_y_test = self.imputer(
            x_train=self.related_context.x[:, prior_idx, :],
            x_test=x_test.repeat(1, self.num_related, 1),
            y_train=self.related_context.y[:, prior_idx])
        imputed_y_test = imputed_y_test[:, prior_idx]
        imputed_y_test = imputed_y_test.reshape(imputed_y_test.shape[0], 1, -1)

        imputed_y = torch.cat([x_train[..., :2], imputed_y],
                              dim=-1)  # prepend the id and fidelity dim
        imputed_y_test = torch.cat([x_test[..., :2], imputed_y_test], dim=-1)

        # fit the model for the current data only as a function of the imputed prior values:
        prior_logits = self.model(
            (
                torch.cat([imputed_y, imputed_y_test]),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            # src_key_padding_mask=padding_mask
        )
        predictions = prior_logits

        if not self.prior_only:
            # collect the meta-unaware target model as well:
            target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
            target_logits = target_model(x_test=x_test)

            # finally convolve the two predictions:
            convolved_logits = DistributionConvolver().to(self.device).convolve(
                A_logits=target_logits,
                borders_A=self.model.criterion.borders,
                B_logits=prior_logits,
                borders_B=self.model.criterion.borders,
                reverse=False,  # we convolve the error model with the prior
                target_borders=self.model.criterion.borders,
                padding=None  # no padding needed here
            )
            predictions = convolved_logits

        self.parent_model.interim_results['logits'] = predictions.cpu()  # for plotting purposes

        # TODO past surprise weighing by prior?

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        return predictions


class JointBatchedMetaContextStrategy(SplitMetaContextStrategy):
    __name__ = "JointBatchedMetaContextStrategy"

    def __init__(self, imputer, D=10, convolve=False, **kwargs):
        """
        D: max tasks per inner batch dimension (fixed at <= 10)
        """
        super().__init__(imputer=imputer, **kwargs)
        self.convolve = convolve

        assert D <= 10, "D must be <= 10 due to PFN input restrictions"
        self.D = D

    def get_prior_augmented_target_model(self, x_train, x_test, y_train):

        num_rel = self.num_related
        # choose B and D such that B*D >= num_rel and D <= self.D
        D = min(self.D, num_rel)
        B = (num_rel + D - 1) // D  # ceil division
        pad = B * D - num_rel

        if pad > 0:
            raise NotImplementedError()

            excess_imputed_y = imputed_y[:, B * D:, :].view(imputed_y.shape[0], -1)
            # todo a separate fwd with different dimensionality on the excess tasks

        # get imputed prior values and pad if needed
        imputed_y = self.parent_model.interim_results['imputed_y'].to(self.device)  # (T, num_rel)
        imputed_y = imputed_y[:, :B * D].view(imputed_y.shape[0], B, D)

        # imputed test ys
        imputed_y_test = self.imputer(
            x_train=self.related_context.x,
            x_test=x_test.repeat(1, num_rel, 1),
            y_train=self.related_context.y
        )  # (T, num_rel)

        imputed_y_test = imputed_y_test[:, :B * D].view(imputed_y_test.shape[0], B, D)

        # prepend the id and fidelity dim of x_train and x_test respectively:
        feature_train = torch.cat([x_train.repeat(1, B, 1), imputed_y], dim=-1)  # (T, B,
        # hp+1)
        feature_test = torch.cat([x_test.repeat(1, B, 1), imputed_y_test],
                                 dim=-1)  # (testT, B, hp+1)

        # now model input is batched across (B,D)
        prior_logits = self.model(
            (
                torch.cat([feature_train, feature_test]),  # (T+testT, B, D)
                y_train.repeat(1, B)  # (T, B)
            ),
            single_eval_pos=y_train.shape[0],
        )

        return prior_logits

    def __call__(self, x_train, x_test, y_train, inc):
        step, _, hp = x_train.shape

        prior_augmented_target_model = partial(
            self.get_prior_augmented_target_model,
            x_train=x_train, y_train=y_train
        )
        prior_logits = prior_augmented_target_model(x_test=x_test)

        target_model = partial(self.get_target_model, x_train=x_train, y_train=y_train)
        target_logits = target_model(x_test=x_test)

        predictions = torch.cat((target_logits, prior_logits), dim=1)

        # compute weights ONCE
        weights = self.parent_model.weights(
            x_train, x_test, y_train, inc, recompute=True
        )

        self.logger.log(
            {'metrics': 'weights', 'step': step,
             **{f'weight_{i}': w.item() for i, w in enumerate(weights)}}
        )

        self.parent_model.interim_results.update({
            'prior_augmented_target_model': prior_augmented_target_model,
            'target_model': target_model,
            'past_x_test': x_test,
            'logits': predictions,
        })

        for callback in self.callbacks:
            callback.on_final_weights(predictions, weights)

        if self.convolve:
            prior_weights = (weights[1:] / weights[1:].sum())  # normalize to sum to 1
            # finally convolve the two predictions:
            convolved_logits, convolved_criterion = DistributionConvolver().to(
                self.device).convolve(
                A_logits=target_logits,
                borders_A=self.model.criterion.borders,
                B_logits=(prior_logits * prior_weights.unsqueeze(1)).sum(dim=1, keepdim=True),
                borders_B=self.model.criterion.borders,
                reverse=False,  # we convolve the error model with the prior
                target_borders=self.model.criterion.borders,
                padding=None  # no padding needed here
            )

            return convolved_logits

        else:
            return (predictions * weights.unsqueeze(1)).sum(dim=1)

        # self.parent_model.interim_results['logits'] = predictions
