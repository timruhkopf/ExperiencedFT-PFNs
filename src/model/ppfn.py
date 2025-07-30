import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from ifbo.transformer import TransformerModel
from model.calc_reliability import calc_target_cv_nll

from src.model.abstractmodel import AbstractModel
from src.utils.dotdict import DotDict
from utils.filelogger import BufferedFileLogger
import pfns4bo

log = logging.getLogger(__name__)

# fixme: move all the plots and logging metrics into an (optional) callback

debug = False


class PPFN(AbstractModel):
    __name__ = "pPFN"

    def __init__(self, model, criterion, logger,
                 related_task_data, min_context_size, imputation_mode='median',
                 incumbent_calculation='imputation-only', flippable_related=False,
                 normalize_to_error_model=True,
                 model_avg='bma',
                 device=None, verbose=True,
                 **kwargs):
        """

        :param model:
        :param criterion:
        :param logger:
        :param mixture_strategy:
        :param related_task_data:
        :param min_context_size:
        :param imputation_mode:
        :param incumbent_calculation: Options: ['imputation-only', 'related-only', 'imputation-and-related']
        :param flippable_related:
        :param device:
        :param verbose:
        """
        self.model: TransformerModel = model if isinstance(model, TransformerModel) else model.model
        self.model.eval()

        # $HOME/anaconda3/envs/ft-pfn-experimental/lib/python3.10/site-packages/pfns4bo/final_models
        # /hebo_morebudget_9_unused_features_3_userpriorperdim2_8.pt.gz
        self.error_model = torch.load(pfns4bo.bnn_model, weights_only=False)
        self.error_model.eval()
        self.error_model.to(device)
        self.original_error_model_borders = deepcopy(self.error_model.criterion.borders)

        self.criterion = criterion if criterion is not None else self.model.criterion

        self.logger = logger
        self.device = device

        self.related_task_data = related_task_data

        self.min_context_size = min_context_size
        self.imputation_mode = imputation_mode
        self.incumbent_calculation = incumbent_calculation
        self.flippable_related = flippable_related
        self.normalize_to_error_model = normalize_to_error_model

        self.model_avg = model_avg

        self.verbose = verbose
        self.kwargs = kwargs

        log.info(f'Instantiated {self.__name__}')

    def _preprocess(self, x_train, y_train, x_test, inc, minimize=True):
        related_context_x = self.related_task_data.x
        related_context_y = self.related_task_data.y
        padding_mask = self.related_task_data.padding_mask

        y_train = y_train.to(self.device)
        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        inc = inc.to(self.device)

        y_train = y_train.unsqueeze(1)
        x_train = x_train.unsqueeze(1)
        x_test = x_test.unsqueeze(1)

        if torch.any(x_test > 999.):
            log.warning(f"Query points x_test contain values > 999: {x_test[x_test > 999.]}")
            x_test = torch.clamp(x_test, max=999.)
        if torch.any(x_train > 999.):
            log.warning(f"Training points x_train contain values > 999: {x_train[x_train > 999.]}")
            x_train = torch.clamp(x_train, max=999.)

        if minimize and self.flippable_related:
            related_context_y = (1 - related_context_y)

        related_context_x = related_context_x.to(self.device)
        related_context_y = related_context_y.to(self.device)

        # (Adjust searchspaces) ------------------------------------------------
        # in case the search spaces are supersets of each other, we need to augment the
        # x_train data to match the related context (we need to drop the dim later for the
        # acquisition function to not notice)
        if x_train.shape[-1] < related_context_x.shape[-1]:
            diff = related_context_x.shape[-1] - x_train.shape[-1]
            placeholder = related_context_x[:, :, -diff:].mean(dim=1).mean(dim=0)
            x_train = torch.cat([
                x_train,
                placeholder.repeat(x_train.shape[0], 1)
            ], dim=-1).to(self.device)

            x_test = torch.cat([
                x_test,
                placeholder.repeat(x_test.shape[0], 1)
            ], dim=-1).to(self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        return x_train, y_train, x_test, inc, \
            related_context_x, related_context_y, padding_mask

    @torch.no_grad()
    def get_ei(self, x_test, inc, x_train=None, y_train=None, minimize=True):
        step = x_train.shape[0]
        x_train, y_train, x_test, inc, \
            related_context_x, related_context_y, padding_mask = \
            self._preprocess(x_train, y_train, x_test, inc, minimize=minimize)

        # (Impute related tasks) -----------------------------------------------
        imputed_y = self.impute(
            related_context_x,
            related_context_y,
            padding_mask,
            x_train
        )

        if step < self.min_context_size:
            # FIXME: EI values!
            return self.warmstart(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc
            )
        else:
            bma_predictions = self.mixture_strategy(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
            )
            # (Collect the PI of the mixture) ----------------------------------
            return self.criterion.ei(
                bma_predictions.squeeze(1), best_f=inc,
                maximize=True)

    @torch.no_grad()
    def get_pi(
            self,
            x_test, inc, x_train=None, y_train=None, minimize=True
    ):
        step = x_train.shape[0]
        x_train, y_train, x_test, inc, \
            related_context_x, related_context_y, padding_mask = \
            self._preprocess(x_train, y_train, x_test, inc, minimize=minimize)

        # plotting anytime performance as a loss
        if debug:

            import matplotlib.pyplot as plt
            import numpy as np

            perf_tensor = 1-(y_train.flatten()) # - related_context_y.max(dim=0).values.cpu(
            # ).numpy())

            # Move to CPU and convert to numpy for ease of processing
            perf_np = perf_tensor.cpu().numpy()

            # Compute the incumbent (best-so-far) performance at each step
            incumbent = np.minimum.accumulate(perf_np)

            # X-axis: time steps
            steps = np.arange(len(incumbent))

            plt.figure(figsize=(8, 5))
            plt.plot(steps, incumbent,  label='Incumbent (Best-so-far)')
            plt.title('Anytime Performance (Incumbent) Over Time')
            plt.xlabel('Time Step')
            plt.ylabel('Performance (Higher is Better)')
            plt.grid(True)
            plt.legend()
            plt.show()

            print('Incumbent performance:', incumbent[-1])
            print('Incumbents of prior tasks',
                  - related_context_y.max(dim=0).values.cpu(
                  ))


        # (Impute related tasks) -----------------------------------------------
        imputed_y = self.impute(
            related_context_x,
            related_context_y,
            padding_mask,
            x_train
        )

        if step < self.min_context_size:
            return self.warmstart(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
                acquisition_fn='pi'
            )
        else:
            bma_predictions = self.mixture_strategy(
                imputed_y,
                related_context_x,
                related_context_y,
                padding_mask,
                x_train,
                x_test,
                y_train,
                inc,
            )
            # (Collect the PI of the mixture) ----------------------------------
            return self.criterion.pi(
                bma_predictions.squeeze(1), best_f=inc,
                maximize=True)

    def impute(
            self, related_context_x, related_context_y, padding_mask, x_train
    ):
        num_related = related_context_x.shape[1]
        # (IMPUTATION to related tasks) ----------------------------------------
        imputed_logits = self.model(
            (
                torch.cat([related_context_x, x_train.repeat(1, num_related, 1)],
                          dim=0),
                torch.cat([related_context_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0],
            # src_key_padding_mask=padding_mask
        )

        if self.imputation_mode == 'median':
            imputed_y = self.criterion.median(imputed_logits)
        elif self.imputation_mode == 'mean':
            imputed_y = self.criterion.mean(imputed_logits)
        elif self.imputation_mode == 'sample':
            # Sample indices from the categorical distributions
            # Shape: (T, n_related_tasks, 1)
            probs = imputed_logits.softmax(-1)
            bins = self.criterion.borders

            # to get the sample at the middle of the bins, we can calculate the middle points
            bucket_middle = (bins[:-1] + bins[:-1] + self.criterion.bucket_widths) / 2

            sampled_indices = torch.stack([
                torch.multinomial(probs[:, i, :], 1) for i in range(probs.shape[1])
            ], dim=1)

            # Remove the last dimension for direct indexing
            # Shape: (T, n_related_tasks)
            sampled_indices = sampled_indices.squeeze(-1)

            # Gather the corresponding bin values
            # Shape: (T, n_related_tasks, num_bars) if bins is 2D, else (T, n_related_tasks)
            imputed_y = bucket_middle[sampled_indices]
        else:
            raise ValueError(f"Unknown imputation mode: {self.imputation_mode}")

        return imputed_y

    def warmstart(
            self,
            imputed_y,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc,
            acquisition_fn='pi'
    ):
        num_related = related_context_x.shape[1]

        if self.incumbent_calculation == 'imputation-only':
            prior_incumbents = imputed_y.max(dim=0).values
        elif self.incumbent_calculation == 'related-only':
            prior_incumbents = related_context_y.max(dim=0).values
        elif self.incumbent_calculation == 'imputation-and-related':
            prior_incumbents = torch.cat([related_context_y, imputed_y, ], dim=0).max(
                dim=0).values
        else:
            raise ValueError(
                f"Unknown incumbent calculation mode: {self.incumbent_calculation}")

        # evaluate the query points under the related tasks augmented with the
        # imputed values at the location of observed points under the target task
        prior_logits = self.model(
            (
                torch.cat([
                    related_context_x,
                    x_train.repeat(1, num_related, 1),
                    x_test.repeat(1, num_related, 1)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        # get the pi at the query points under the related tasks,
        prior_incumbents = prior_incumbents.unsqueeze(1).repeat(1, x_test.shape[0])
        acq_fn = getattr(self.model.criterion, acquisition_fn)
        acq_related = torch.stack([
            acq_fn(
                prior_logits[:, b, :].squeeze(1),
                best_f=prior_incumbents[b, :].unsqueeze(1),
                maximize=True
            )
            for b in range(num_related)
        ], dim=0)

        # get the pi under the target task
        target_logits = self.model(
            (
                torch.cat([x_train, x_test], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],

        )
        acq_target = acq_fn(
            target_logits.squeeze(1), best_f=inc,
            maximize=True
        )

        # here we want to be maximally aggressive from the perspective of the priors,
        # and encourage exploring successful incumbents under the related tasks
        return torch.cat([acq_target.unsqueeze(0), acq_related], dim=0).max(axis=0).values

    def mixture_strategy(
            self,
            imputed_y,
            related_context_x,
            related_context_y,
            padding_mask,
            x_train,
            x_test,
            y_train,
            inc
    ):
        import torch  # fixme: why is torch otherwise not detected ?
        step = x_train.shape[0]
        num_related = related_context_x.shape[1]
        # TODO the following two forwards can be batched together with
        #  appropriate padding masks. this will save wallclock time
        # (Collect target task logits) --------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        target_logits = self.model(
            (
                torch.cat([
                    x_train,

                    # Query
                    x_test,
                    x_train if self.verbose == True else torch.tensor([]).to(self.device)
                ], dim=0),
                y_train
            ),
            single_eval_pos=x_train.shape[0],
            src_key_padding_mask=None
        )

        # (Collect prior logits) -------------------------------------------
        # CAREFUL: here we also do the query forward for the evidence
        prior_logits = self.model(
            (
                torch.cat([
                    # train
                    related_context_x,
                    x_train.repeat(1, num_related, 1),

                    # Query
                    x_test.repeat(1, num_related, 1),
                    related_context_x,  # we need this to project the prior into the target task
                    # later
                    x_train.repeat(1, num_related, 1) if self.verbose == True else torch.tensor([]).to(self.device)
                ], dim=0),
                torch.cat([related_context_y, imputed_y, ], dim=0)
            ),
            single_eval_pos=related_context_x.shape[0] + x_train.shape[0],
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        if self.model_avg == 'ppd_mixture_eqw':
            query_size = x_test.shape[0]
            logits = torch.concat([prior_logits[:query_size], target_logits[:query_size]], dim=1)
            prediction = logits.mean(dim=1)
            return prediction

        # (Collect difference function) ------------------------------------
        # we calcualte the difference function between the related tasks and target task
        # anchored in the imputed values. Since we are only interested in the

        # The error model has a power projection type of non-uniform binning.
        # to have the error model's maximal resolution, we scale the residuals
        # to an interval [-2, 2] and undo this later!
        target_borders = self.criterion.borders
        error_borders = self.error_model.criterion.borders

        y_target = y_train.repeat(1, num_related) - imputed_y

        if self.normalize_to_error_model:
            # Here we exaggerate the difference to meet the high resolution range [-2.5, 2.5]
            # of the model -- we will need to undo this later to communicate the
            # result distribution in the target binning for convolution.
            bandwidths = self.error_model.criterion.bucket_widths

            if debug:
                import numpy as np
                import matplotlib.pyplot as plt

                probabilities = bandwidths.numpy()
                probabilities /= probabilities.sum()  # normalize if not already

                # CDF calculation: cumulative sum
                cdf = np.concatenate([[0], np.cumsum(probabilities)])

                plt.figure(figsize=(8, 4))
                plt.step(error_borders, cdf, where='post',
                         label="CDF")
                plt.xlabel("Value")
                plt.ylabel("Cumulative Probability")
                plt.title("Cumulative Distribution Function (CDF)")
                plt.grid(True)
                plt.legend()
                plt.show()

            # now we determine the factor by whcih we need to scale y_target, such that
            # we do not exceed the error model's resolution range
            FACTOR = 2.5 / max(y_target.abs().max(), 1e-6)  # avoid division by zero

            y_error = FACTOR * y_target
            if debug:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Convert tensors to numpy arrays
                y_target_np = y_target.numpy().flatten()
                y_error_np = y_error.numpy().flatten()
                error_borders_np = error_borders.numpy().flatten()

                plt.figure(figsize=(8, 6))

                # Plot overlapping histograms
                plt.hist(y_target_np, bins=30, alpha=0.5, label='y_target', color='blue',
                         edgecolor='black')
                plt.hist(y_error_np, bins=30, alpha=0.5, label='y_error', color='red',
                         edgecolor='black')

                # Add rug plot for error_borders
                sns.rugplot(error_borders_np, color='green', height=0.05)

                plt.title('Overlapping Histograms with Rug plot of error_borders')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.legend()

                plt.show()

        else:
            y_error = y_target

        error_logits = self.error_model(
            (
                torch.cat([
                    # NOTICE: we need to crop the idx dim in ifbo (ifbo paper section 5.2),
                    # otherwise the rmse will increase in predicting the idx!
                    x_train[:, :, 1:].repeat(1, num_related, 1),

                    # Query
                    x_test[:, :, 1:].repeat(1, num_related, 1),
                    related_context_x[:, :, 1:],  # for the projection of the prior into the target
                    x_train[:, :, 1:].repeat(1, num_related, 1) if self.verbose == True else
                    torch.tensor([]).to(self.device)
                ], dim=0),
                y_error
            ),
            single_eval_pos=x_train.shape[0],
            # fixme: this model is not capable of accepting padding masks yet!
            # src_key_padding_mask=torch.cat([
            #     padding_mask,
            #     torch.zeros(num_related, x_train.shape[0], dtype=torch.bool).to(self.device)
            # ], dim=1)
        )

        if self.verbose and debug:  # FIXME: this metric needs x_train as query in both the prior
            # and error_logits"!
            imputation_diffs = y_train.repeat(1, num_related) - imputed_y

            if debug:
                import torch
                import matplotlib.pyplot as plt

                # Imputation differences histogram at the current time step
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.flatten()

                for i in range(imputation_diffs.shape[1]):
                    data = imputation_diffs[:, i].numpy()  # convert tensor column to numpy
                    axes[i].hist(data, bins=20, edgecolor='black')
                    axes[i].set_title(f'Histogram of imputation_diffs column {i + 1}')
                    axes[i].set_xlabel(f'Column {i + 1} values')
                    axes[i].set_ylabel('Frequency')

                plt.tight_layout()
                plt.show()

            imputation_diffs = imputation_diffs.mean(dim=0)
            self.logger.log({
                'metrics': 'imputation_diff',
                'step': step,
                **{f'imputation_diff{i}': imputation_diffs[i].item()
                   for i in range(imputation_diffs.shape[0])}
            })
            # Plotting the average imputation differences over time
            if debug:
                import pandas as pd
                import matplotlib.pyplot as plt
                df = pd.DataFrame(self.logger.logs)
                df = df[df['metrics'] == 'imputation_diff']
                cols = list(df.columns[df.columns.str.startswith('imputation_diff')]) + ['step']
                subset = df.loc[:, cols]
                subset.plot(x='step')
                plt.show()

            y_hat = self.error_model.criterion.median(
                error_logits[-x_train.shape[0]:]  # y_train error logits!
            )

            if debug:
                import matplotlib.pyplot as plt
                import seaborn as sns

                # Convert tensors to numpy arrays
                y_target_np = y_target.numpy().flatten()
                y_hat_np = y_hat.numpy().flatten() / 16  # for median we can just divide!
                error_borders_np = error_borders.numpy().flatten()

                plt.figure(figsize=(8, 6))

                # Plot overlapping histograms
                plt.hist(y_target_np, bins=30, alpha=0.5, label='y_target', color='blue',
                         edgecolor='black')
                plt.hist(y_hat_np, bins=30, alpha=0.5, label='y_hat_median', color='red',
                         edgecolor='black')

                # Add rug plot for error_borders
                sns.rugplot(error_borders_np, color='green', height=0.01)

                plt.title('Overlapping Histograms with Rug plot of error_borders')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.legend()

                plt.show()

            # unprojected rmse! (i.e. in the error model's criterion borders)
            y = y_error
            rmse = torch.sqrt(torch.mean((y - y_hat) ** 2, dim=0))

            self.logger.log({
                'metrics': 'rmse',
                'step': step,
                **{f'rmse_{i}': rmse[i].item()
                   for i in range(rmse.shape[0])}
            })

            # Plotting the error_model's RMSE over time
            if debug:
                import pandas as pd
                import matplotlib.pyplot as plt
                df = pd.DataFrame(self.logger.logs)
                df = df[df['metrics'] == 'rmse']
                cols = list(df.columns[df.columns.str.startswith('rmse')]) + ['step']
                subset = df.loc[:, cols]
                subset.plot(x='step')
                plt.show()

            # Plotting the predicted differences in 3d (1d hp + 1 fidelity dim)
            if debug:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

                import numpy as np
                import plotly.graph_objs as go
                from plotly.offline import plot

                # find the location in y where the inc lives
                inc_y = inc[0]
                inc_x = x_train[(max(y_train) == y_train).view(-1), 0,
                        1:].numpy()  # assuming x_test is
                # of shape (T, 1, D)

                # Convert torch tensors to numpy arrays as before
                x_related = related_context_x[:, :, 1:].numpy()
                y_related = related_context_y.numpy()
                x_np = x_train[:, 0, 1:].numpy()
                y_train_np = y_train[:, 0].numpy()
                imputed_y_np = imputed_y[:, 0].numpy()
                y_hat_np = y_hat[:, 0].numpy()

                print('inc in train', max(y_train), 'incumbent y', inc_y, 'incumbent x', inc_x)
                # Scatter traces
                trace_y_train = go.Scatter3d(
                    x=x_np[:, 0], y=x_np[:, 1], z=y_train_np,
                    mode='markers',
                    marker=dict(color='blue', size=5),
                    name='y_train'
                )

                # plot the incumbent point
                trace_incumbent = go.Scatter3d(
                    x=inc_x[:, 0], y=inc_x[:, 1], z=inc_y,
                    mode='markers',
                    marker=dict(color='black', size=5, symbol='x'),
                    name='incumbent'
                )

                trace_imputed_y = go.Scatter3d(
                    x=x_np[:, 0], y=x_np[:, 1], z=imputed_y_np,
                    mode='markers',
                    marker=dict(color='green', size=5, symbol='diamond'),
                    # Use a supported symbol here
                    name='imputed_y'
                )

                trace_y_hat = go.Scatter3d(
                    x=x_np[:, 0], y=x_np[:, 1], z=y_hat_np,
                    mode='markers',
                    marker=dict(color='red', size=5),
                    name='y_hat'
                )

                trace_related = go.Scatter3d(
                    x=x_related[:, 0, 0], y=x_related[:, 0, 1], z=y_related[:, 0],
                    mode='markers',
                    marker=dict(color='orange', size=5),
                    name='related_context_y'
                )

                # Lines from y_train to y_hat (difference vectors)
                line_traces = []
                for i in range(len(x_np)):
                    line_traces.append(
                        go.Scatter3d(
                            x=[x_np[i, 0], x_np[i, 0]],
                            y=[x_np[i, 1], x_np[i, 1]],
                            z=[y_train_np[i] - y_hat_np[i], y_train_np[i]],
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False
                        )
                    )

                data = ([
                            trace_y_train,
                            trace_imputed_y,
                            trace_y_hat,
                            trace_related,
                            trace_incumbent
                        ] \
                        + line_traces
                        )

                layout = go.Layout(
                    scene=dict(
                        xaxis_title='X dimension 1',
                        yaxis_title='X dimension 2',
                        zaxis_title='Y values'
                    ),
                    title='Interactive 3D plot showing y_train and y_hat differences',
                    legend=dict(x=0, y=1),
                    margin=dict(l=0, r=0, b=0, t=40)
                )

                fig = go.Figure(data=data, layout=layout)

                # This will open the interactive plot in your default web browser
                plot(fig)

        # (Project prior logits into target task) --------------------------
        # Here we take the predicted prior logits of the x_test and need to adjust them
        # according to the error logits. -- which tell us how to shift the distribution
        # (median) and given the shift, how to adjust the probability mass.
        # given the prior logits and error logits are differently binned distributions
        # we need to interpret the error bins and adjust probabiltiy mass of the prior logits

        # This projection can be done by computing the fractional overlap of each original bin
        # with each common bin and distributing the original bin’s probability accordingly.
        # now let us move the error logits into the prior logits space
        # normalize the error borders to the target borders range:
        if self.normalize_to_error_model:

            self.error_model.criterion.borders = self.original_error_model_borders / FACTOR
            # scaling here affects the size of the kernel and the cost of the conv.
            left = min(self.error_model.criterion.icdf(error_logits[:, 0, :], 0.01))
            right = max(self.error_model.criterion.icdf(error_logits[:, 0, :], 0.99))

            kernel_grid = torch.arange(
                left, right,
                step=(target_borders[1] - target_borders[0]).item()
            )
            error_probs_kernel = project_probs_to_common_bins_batch(
                F.softmax(error_logits, dim=-1),
                self.error_model.criterion.borders,
                kernel_grid
            )

        else:
            error_probs_kernel = F.softmax(error_logits, dim=-1)
            kernel_grid = self.criterion.borders

        # now we need to convolve the prior_probs with the error probs -------
        # this will provide us with the projection of the prior tasks into the target task
        # domain.
        # flatten the time and batch dimensions for convolution
        prior_probs = F.softmax(prior_logits, dim=-1)
        T, B, D = prior_probs.shape
        convolved_logits = convolve_probs_with_error(
            # probs, probs_bins, kernel, kernel_bins
            probs=prior_probs.view(-1, D),
            probs_bins=target_borders,
            kernel=error_probs_kernel.view(-1, error_probs_kernel.shape[-1]),
            kernel_bins=kernel_grid

        ).reshape(T, B, -1)

        # now the convolved logits describe:
        # x_test, related_context_x, (and if debug=True x_train) in the target task space

        # (Bayesian model averaging) -----------
        # prior_predictions = convolved_logits[:-step]

        # p(y|M_i) = p(y|M_i, D) p(D|M_i) but as logits!
        # predictions = torch.concat([target_logits, prior_predictions], dim=1).to(self.device)

        query = x_test.shape[0]
        predictions = torch.cat(
            [target_logits[:query], convolved_logits[:query]], dim=1
        ).to(self.device)

        if self.model_avg == 'project_eqw':  # equally weighted average
            # here we simply average the predictions over the related tasks
            return predictions.mean(dim=1)

        if self.model_avg in ['prior-mixture', 'bma', 'bma-decay', 'bma-cv-target']:
            # To determine p(M | H) = p(H | M) p(M) / p(H)
            # we need to calculate the evidence p(H | M_i) for each model M_i
            # here we calculate only for the related tasks.

            # first project the x related data into the target task space via the error model
            # remember the convolved logits are the prior convolved with error logits at that pos,
            related_context_y_hat = convolved_logits[query:query + related_context_x.shape[0]]
            # associated with the related_context_x

            # the pfn cannot take in distributions, so we need to take one point
            # TODO consider sampling here!
            related_context_y_hat_median = self.criterion.median(related_context_y_hat)

            # now we can learn a model of the projected prior data in the target space
            projected_prior_experience = self.model(
                (
                    torch.cat(
                        [
                            # train
                            related_context_x,
                            # query
                            x_train.repeat(1, num_related, 1),
                        ]
                    ),
                    related_context_y_hat_median
                ),
                single_eval_pos=related_context_x.shape[0],
            )

            # finally we evaluate the prior target data predictions against the
            # observed y_train values, telling us how well the projected prior data
            # would explain the observed data. This gives us the relative weight of each prior
            prior_evidence = torch.stack([
                self.criterion(projected_prior_experience[:, b, :].squeeze(1), y_train)
                for b in range(num_related)
            ], dim=0).to(self.device).mean(dim=1)

            prior_weights = torch.softmax(-prior_evidence, dim=-1)
            self.logger.log(
                {'metrics': 'prior_weights', 'step': step,
                 **{f'prior_weight_{i}': w.item()
                    for i, w in enumerate(prior_weights)}},
            )

            if debug:
                import matplotlib.pyplot as plt
                df = self.logger.df[self.logger.df['metrics'] == 'prior_weights']
                df.set_index('step')
                df = df[[col for col in df.columns if
                         col.startswith('bma_weight')]]

                df.plot(figsize=(10, 6))
                plt.xlabel('Step')
                plt.ylabel('Weight')
                plt.title('Time Series of Prior PPD BMA Weights')
                plt.grid(True)
                plt.show()

            # given the prior weights, we can now calculate the weighted average of the
            # prior predictions:
            # p(y|.) = \sum_i  p(y|M_i) p(M_i | D)
            prior_prediction = (predictions[:, 1:] * prior_weights.unsqueeze(-1)).sum(dim=1)

            if self.model_avg == 'prior-mixture':
                # here we simply average the predictions over the related tasks
                return prior_prediction

        if self.model_avg == 'bma-decay':
            # In this case we do
            # \alpha(step) p(y|M_{\tau^*}) + (1-\alpha(step)) p_BMA(y|{M_\tau}_{\tau
            # \neq \tau^*})
            # where \tau^* is the target task and \alpha(step) is an exponential decay function
            # that decays with the number of steps.

            alpha = constant_exponential(
                step,
                lambda_=self.kwargs.get('decay_rate', 0.003),
                constant=self.min_context_size
            )

            if debug:
                min_context_size = 10  # constant parameter
                lambda_ = 0.003  # decay rate
                steps = np.arange(0, 1001)  # 0-1000 inclusive
                alpha = constant_exponential(steps, lambda_, min_context_size)
                plt.figure(figsize=(10, 6))
                plt.plot(steps, alpha)
                plt.title('Alpha decay over steps')
                plt.xlabel('Step')
                plt.ylabel('Alpha')
                plt.grid(True)
                plt.show()

            return alpha * predictions[:, 0] + (1 - alpha) * prior_prediction

        if self.model_avg == 'bma-cv-target':
            assert x_train.shape[0] >= 10, \
                "BMA with cross-validation requires at least 10 training points."
            target_nll = calc_target_cv_nll(
                x_train.squeeze(1),
                y_train.squeeze(1),
                self.model,
                self.criterion,
                splits=5,
                random_state=42
            ).unsqueeze(0).to(self.device)

            weights = torch.softmax(torch.cat([-target_nll, -prior_evidence]), dim=-1)
            self.logger.log(
                {'metrics': 'prior_weights', 'step': step,
                 **{f'prior_weight_{i}': w.item()
                    for i, w in enumerate(weights)}},
            )
            if debug:
                import matplotlib.pyplot as plt
                df = self.logger.df[self.logger.df['metrics'] == 'prior_weights']
                df.set_index('step')
                df = df[[col for col in df.columns if
                         col.startswith('bma_weight')]]

                df.plot(figsize=(10, 6))
                plt.xlabel('Step')
                plt.ylabel('Weight')
                plt.title('Time Series of Prior PPD BMA Weights')
                plt.grid(True)
                plt.show()

            return (predictions * weights.unsqueeze(-1)).sum(dim=1)

        if self.model_avg == 'bma':
            raise NotImplementedError(
                "BMA model averaging is not implemented yet, "
                "we didn't find a way to calculate the evidence p(H | M_{\\tau^*})."
            )


def constant_exponential(n_target, lambda_=0.001, constant=0):
    effective_n = torch.maximum(torch.tensor(n_target - constant, dtype=torch.float32),
                                torch.tensor(0.0))
    return 1 - torch.exp(-lambda_ * effective_n)


def convolve_probs_with_error(probs, probs_bins, kernel, kernel_bins):
    batch_size, length = probs.shape
    kernel_size = kernel.shape[1]

    # Original input shape: (batch_size, 1, length)
    p_orig_t = probs.view(batch_size, 1, length)

    # Flip kernels for convolution
    p_shift_flipped = torch.flip(kernel, dims=[1]).view(batch_size, 1, kernel_size)

    # Now, merge batch into channels dimension by transposing:
    # Input: (batch_size, 1, length) -> (1, batch_size, length)
    p_orig_t_merged = p_orig_t.permute(1, 0, 2)  # (1, batch_size, length)

    # Weight already has shape (batch_size, 1, kernel_size)
    # To match input's channels, reshape kernels as (batch_size, 1, kernel_size)
    # Perform conv1d with groups = batch_size
    p_convolved = F.conv1d(
        p_orig_t_merged,  # input channels == batch_size
        p_shift_flipped,
        # weight shape must be (out_channels, in_channels/groups, kernel_size); here out_channels=batch_size, in_channels/groups=1
        padding=kernel_size - 1,
        groups=batch_size
    )

    # p_convolved shape: (1, batch_size, output_length)
    # reshape back to (batch_size, output_length)
    p_convolved = p_convolved.permute(1, 0, 2).view(batch_size, -1)
    p_convolved /= p_convolved.sum(dim=1, keepdim=True)  # probability norm

    # calculate the new bins of the convolved distribution
    min_kernel = kernel_bins[0]
    max_kernel = kernel_bins[-1]
    step = probs_bins[1] - probs_bins[0]
    prob_min = probs_bins[0]
    prob_max = probs_bins[-1]
    L = probs.shape[1]
    K = kernel.shape[1]
    conv_len = L + K - 1

    # Construct bin edges for original, kernel, and convolved (centered bins)
    # bins_probs = np.arange(prob_min, prob_max + step, step)  # Should have length L
    # bins_kernel = np.arange(min_kernel, max_kernel + step, step)  # length K
    bins_convolved = torch.arange(
        prob_min + min_kernel,
        prob_max + max_kernel + step,
        step
    )  # length conv_len

    # Calculate start and end indices for cropping (center-crop)
    start = (kernel_size - 1) // 2
    end = start + L

    # Crop convolved output & adjust the probability mass by adding the missing mass
    # to the left and right edges
    p_convolved_cropped = p_convolved[:, start:end]
    missing_prob_right = p_convolved[:, end:].sum(dim=1)
    missing_prob_left = p_convolved_cropped[:, :start].sum(dim=1)
    p_convolved_cropped[:, start] += missing_prob_left
    p_convolved_cropped[:, -1] += missing_prob_right

    debug = False
    if debug:
        # Illustration of the convolved distribution to verify implementation
        import numpy as np
        import matplotlib.pyplot as plt

        idx = 9

        # Convert tensors to numpy for plotting
        probs_0 = probs[idx].numpy()
        kernel_0 = kernel[idx].numpy()
        p_convolved_0 = p_convolved[idx].numpy()
        p_convolved_cropped_0 = p_convolved_cropped[idx].numpy()

        # bin centers for plotting
        centers_probs_bins = (probs_bins[:-1] + probs_bins[1:]) / 2
        centers_kernel_bins = (kernel_bins[:-1] + kernel_bins[1:]) / 2
        centers_bins_convolved = (bins_convolved[:-1] + bins_convolved[1:]) / 2

        plt.figure(figsize=(12, 7))
        plt.plot(centers_probs_bins, probs_0, label='Original probs')
        plt.plot(centers_kernel_bins, kernel_0, label='Error probs (kernel)')
        plt.plot(centers_bins_convolved[:-2].numpy(), p_convolved_0, label='Full Convolved')

        cropped_bins = centers_bins_convolved[start:end]
        plt.plot(cropped_bins, p_convolved_cropped_0, label='Cropped Convolved')

        plt.title('Comparison of Original, Kernel, and Convolved Distributions')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.xlim([bins_convolved[0], bins_convolved[-1]])
        plt.show()

    logits_convolved = torch.log(p_convolved_cropped.clamp(min=1e-12))

    return logits_convolved


# FIXME: double check the implementation of the logit projection
def project_probs_to_common_bins_batch(orig_probs, orig_bounds, target_bounds):
    """
    Project batched probability distributions defined on orig_bounds to target_bounds 
    by fractional overlap of bins in PyTorch.

    Args:
        orig_probs   : Tensor of shape (..., N) -- probabilities over original bins.
        orig_bounds  : 1D tensor of length N+1 -- bin edges of original distribution.
        target_bounds: 1D tensor of length M+1 -- bin edges of target distribution.

    Returns:
        projected_probs : Tensor of shape (..., M) -- probabilities projected onto target bins.
    """
    # orig_probs: (..., N)
    orig_shape = orig_probs.shape
    N = orig_shape[-1]
    M = target_bounds.shape[0] - 1

    # Expand bins for vectorized overlap calculation:
    # orig lefts and rights shape: (N, 1)
    orig_lefts = orig_bounds[:-1].unsqueeze(1)  # (N,1)
    orig_rights = orig_bounds[1:].unsqueeze(1)  # (N,1)
    # target lefts and rights shape: (1, M)
    target_lefts = target_bounds[:-1].unsqueeze(0)  # (1, M)
    target_rights = target_bounds[1:].unsqueeze(0)  # (1, M)

    # Calculate overlaps (N x M)
    overlaps = torch.clamp(
        torch.min(orig_rights, target_rights) - torch.max(orig_lefts, target_lefts),
        min=0.0)  # (N,M)
    orig_widths = (orig_rights - orig_lefts)  # (N,1)
    fractions = overlaps / orig_widths  # (N,M)

    # Move orig_probs last dim (N) to front to do batch matmul:
    # orig_probs reshaped to (-1, N)
    orig_probs_flat = orig_probs.reshape(-1, N)  # (B, N)

    # Multiply: (B, N) @ (N, M) => (B, M)
    projected_flat = torch.matmul(orig_probs_flat, fractions)  # (B, M)

    # Normalize so projected probabilities sum to 1 (for each batch)
    projected_flat /= projected_flat.sum(dim=1, keepdim=True)

    # Reshape back to original batch dims + M
    projected_shape = orig_shape[:-1] + (M,)
    projected_probs = projected_flat.reshape(projected_shape)

    debug = False
    if debug:
        plt.hist(projected_probs[0, 0].numpy(), bins=target_bounds, alpha=0.5,
                 label='Projected Probs')
        plt.show()

    return projected_probs


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # Original probability distribution and bin edges
    orig_probs = np.array([0.05, 0.15, 0.3, 0.2, 0.1, 0.2])
    orig_bounds = np.array([0, 1, 2, 3, 4, 5, 6])  # 6 bins

    # Target bin edges (non-uniform widths)
    target_bounds = np.array([0, 0.5, 2.5, 3, 4.5, 6])  # 5 bins

    N = len(orig_probs)
    M = len(target_bounds) - 1

    # Step 1: Compute overlaps between each original and target bin
    orig_lefts = orig_bounds[:-1][:, None]  # (N, 1)
    orig_rights = orig_bounds[1:][:, None]  # (N, 1)
    target_lefts = target_bounds[:-1][None, :]  # (1, M)
    target_rights = target_bounds[1:][None, :]  # (1, M)

    # Overlap lengths for each (orig_bin, target_bin) pair
    overlaps = np.clip(
        np.minimum(orig_rights, target_rights) - np.maximum(orig_lefts, target_lefts),
        0, None
    )  # shape (N, M)

    orig_widths = orig_rights - orig_lefts  # (N, 1)
    fractions = overlaps / orig_widths  # (N, M)

    # Step 2: Redistribute probabilities using the fractions matrix
    # (orig_probs shape (N,), fractions (N, M))
    projected_probs = orig_probs @ fractions  # shape (M,)

    # Step 3: Normalize (optional—should already sum to 1, but for safety)
    projected_probs /= projected_probs.sum()

    # --- Visualization ---
    bin_centers_orig = (orig_bounds[:-1] + orig_bounds[1:]) / 2
    bin_centers_proj = (target_bounds[:-1] + target_bounds[1:]) / 2

    plt.figure(figsize=(8, 5))

    # Plot original histogram
    plt.bar(bin_centers_orig, orig_probs, width=1, alpha=0.7, label='Original', color='royalblue',
            edgecolor='black')

    # Plot projected histogram (shifted a bit for clarity)
    widths_proj = target_bounds[1:] - target_bounds[:-1]
    plt.bar(target_bounds[:-1], projected_probs,
            width=widths_proj,
            align='edge',
            alpha=0.6,
            label='Projected',
            color='orange',
            edgecolor='black')

    # Draw original and target bin edges
    for b in orig_bounds:
        plt.axvline(b, color='blue', ls='--', lw=1, alpha=0.25)
    for b in target_bounds:
        plt.axvline(b, color='orange', ls=':', lw=1, alpha=0.5)

    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Redistribution of Histogram Probabilities to New Bins')
    plt.tight_layout()
    plt.show()
