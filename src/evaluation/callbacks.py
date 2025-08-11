import abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from plotly.subplots import make_subplots
from plotly.offline import plot

import plotly.graph_objects as go


class AbstractCallback(abc.ABC):
    __name__ = 'AbstractCallback'
    DEBUG = False
    STOP_AT = -1

    def __init__(self, model, error_model, counterfit, related_context, logger, device, **kwargs):
        self.logger = logger
        self.device = device
        self.model = model
        self.error_model = error_model
        self.counterfit = counterfit  # 'median' or 'mc'
        self.related_context = related_context
        self.num_related = related_context.x.shape[1]

        self.borders_model = self.model.criterion.borders
        self.borders_error = self.error_model.criterion.borders

    def step(self, x_train):
        return x_train.shape[0]

    def on_acq_start(self, x_train, y_train, x_test, inc):
        pass

    def on_acq_end_warmstart(self, x_train, y_train, x_test, inc, pi_values):
        pass

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        pass

    def on_acq_end_mixture(self, x_train, y_train, x_test, inc, predictions):
        pass


class CallbackAnytime(AbstractCallback):
    __name__ = 'CallbackAnytime'

    def on_acq_start(self, x_train, y_train, x_test, inc):
        if self.DEBUG and self.step(x_train) == self.STOP_AT:
            self.plot_anytime(y_train)

    def plot_anytime(self, y_train):
        perf_tensor = 1 - (y_train.flatten())  # - related_context_y.max(dim=0).values.cpu(
        # ).numpy())

        # Move to CPU and convert to numpy for ease of processing
        perf_np = perf_tensor.cpu().numpy()

        # Compute the incumbent (best-so-far) performance at each step
        incumbent = np.minimum.accumulate(perf_np)

        # X-axis: time steps
        steps = np.arange(len(incumbent))

        plt.figure(figsize=(8, 5))
        plt.plot(steps, incumbent, label='Incumbent (Best-so-far)')
        plt.title('Anytime Performance (Incumbent) Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Performance (Higher is Better)')
        plt.grid(True)
        plt.legend()
        plt.show()

        print('Incumbent performance:', incumbent[-1])
        print('Incumbents of prior tasks',
              - self.related_task_data.y.max(dim=0).values.cpu(
              ))


class CallbackImputationDiff(AbstractCallback):
    __name__ = 'CallbackImputationDiff'

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        # y_error = rescaled imputation diff!
        imputation_diffs = y_train.repeat(1, self.num_related) - imputed_y

        if self.DEBUG and self.step(x_train) == self.STOP_AT:
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

            # Plotting the average imputation differences over time
            import pandas as pd
            import matplotlib.pyplot as plt
            df = pd.DataFrame(self.logger.logs)
            df = df[df['metrics'] == 'imputation_diff']
            cols = list(df.columns[df.columns.str.startswith('imputation_diff')]) + ['step']
            subset = df.loc[:, cols]
            subset.plot(x='step')
            plt.show()

        imputation_diffs = imputation_diffs.mean(dim=0)
        self.logger.log({
            'metrics': 'imputation_diff',
            'step': self.step(x_train),
            **{f'imputation_diff{i}': imputation_diffs[i].item()
               for i in range(imputation_diffs.shape[0])}
        })


class CallbackErrorModelRMSE(AbstractCallback):
    __name__ = 'CallbackErrorModelRMSE'

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, projected_logits, imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        imputation_diffs = y_train.repeat(1, self.num_related) - imputed_y

        y_hat = self.error_model.criterion.median(
            error_logits[-x_train.shape[0]:]  # y_train error logits!
        )
        # y_error = rescaled imputation diff!
        rmse = torch.sqrt(torch.mean((y_error - y_hat) ** 2, dim=0))

        self.logger.log({
            'metrics': 'rmse',
            'step': self.step(x_train),
            **{f'rmse_{i}': rmse[i].item()
               for i in range(rmse.shape[0])}
        })

    def plot(self):
        df = pd.DataFrame(self.logger.logs)
        df = df[df['metrics'] == 'rmse']
        cols = list(df.columns[df.columns.str.startswith('rmse')]) + ['step']
        subset = df.loc[:, cols]
        subset.plot(x='step')
        plt.show()


class Callback1DProjection(AbstractCallback):
    __name__ = 'Callback1DProjection'

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        import plotly.graph_objs as go
        from plotly.offline import plot

        # find the location in y where the inc lives
        inc_y = inc[0]
        inc_x = x_train[(max(y_train) == y_train).view(-1), 0,
        1:].numpy()  # assuming x_test is
        # of shape (T, 1, D)

        y_hat = self.error_model.criterion.median(
            error_logits[-x_train.shape[0]:]  # y_train error logits!
        )

        # Convert torch tensors to numpy arrays as before
        x_related = self.related_context.x[:, :, 1:].numpy()
        y_related = self.related_context.y.numpy()
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


class Callback1dProjectionPI(AbstractCallback):
    __name__ = 'Callback1DProjectionPI'
    STOP_AT = 20

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, projected_logits, imputed_y, y_error,
            x_train, y_train, x_test, inc
    ):
        self.target_logits = target_logits
        self.prior_logits = prior_logits
        self.error_logits = error_logits
        self.projected_logits = projected_logits
        self.imputed_y = imputed_y
        self.y_error = y_error
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.inc = inc

    def on_final_weights(self, predictions, weights):
        self.predictions = predictions
        self.weights = weights

    def on_acq_end_mixture(self, x_train, y_train, x_test, inc, predictions):
        if self.step(x_train) == self.STOP_AT:
            def get_lower_upper_surfaces(model, logits, x1, x2, fig, row, col, grid_size,
                                         name='', ):
                lower = model.criterion.icdf(logits, 0.25).numpy().reshape(-1)
                mean = model.criterion.mean(logits).numpy().reshape(-1)
                upper = model.criterion.icdf(logits, 0.75).numpy().reshape(-1)

                # Lower quantile
                fig.add_trace(go.Surface(
                    x=x1, y=x2, z=lower.reshape(grid_size, grid_size), colorscale='Oranges',
                    opacity=0.5,
                    showscale=False, coloraxis=None
                    # name=f'25th Q {name}'
                ),
                    row=row, col=col
                )

                # Upper quantile
                fig.add_trace(go.Surface(
                    x=x1, y=x2, z=upper.reshape(grid_size, grid_size), colorscale='Oranges',
                    opacity=0.7,  showscale=False, coloraxis=None
                    #             name=f'75th Q {name}'
                ),
                    row=row, col=col
                )

            # Create subplot figure with 2 rows and 2 columns
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=("Target", "final prediction", "Error", "plot4", "Raw Prior",
                                "Projected Prior",),
                specs=[[{"type": "surface"}, {"type": "surface"}],
                       [{"type": "surface"}, {"type": "surface"}],
                       [{"type": "surface"}, {"type": "surface"}]] , # specify plot types per
                # subplot

                vertical_spacing=0.03,  # default is ~0.3, smaller makes rows tighter
                horizontal_spacing=0.03  # default is ~0.2, smaller makes
            )

            # make a meshgrid for the x_grid
            grid_size = 50
            x1 = np.linspace(0, 1, grid_size)
            x2 = np.linspace(0, 1, grid_size)
            X1, X2 = np.meshgrid(x1, x2)
            config_id = torch.tensor((X1 * 999).ravel()).floor()
            X_grid = torch.vstack([config_id, torch.tensor(X1.ravel()), torch.tensor(X2.ravel())]).T
            X_grid = X_grid.to(self.device).float().unsqueeze(1)



            # TARGET MODEL -----------------------------------------------------
            target_logits = self.model(
                (
                    torch.cat([
                        x_train,
                        X_grid
                    ]),
                    y_train
                ),
                single_eval_pos=x_train.shape[0]
            )

            get_lower_upper_surfaces(self.model, target_logits, X1, X2, fig, row=1, col=1,
                                     name='target', grid_size=grid_size)

            # add the scatter plot for x_train, y_train
            fig.add_trace(go.Scatter3d(
                x=x_train[:, 0, 1].cpu().numpy(),
                y=x_train[:, 0, 2].cpu().numpy(),
                z=y_train[:, 0].cpu().numpy(),
                mode='markers', marker=dict(size=2, color='red'),
                name='Training Points',
            ), row=1, col=1)

            # PRIOR MODEL -----------------------------------------------------
            prior_logits = self.model(
                (
                    torch.cat([
                        self.related_context.x,
                        X_grid.repeat(1, self.num_related, 1)
                    ]),
                    self.related_context.y
                ),
                single_eval_pos=self.related_context.x.shape[0]
            )

            get_lower_upper_surfaces(self.model, prior_logits, X1, X2, fig, row=3, col=1,
                                     name='prior', grid_size=grid_size)

            # PRIOR DATA ------
            # add the scatter plot for x_train, y_train
            fig.add_trace(go.Scatter3d(
                x=self.related_context.x[:, 0, 1].cpu().numpy(),
                y=self.related_context.x[:, 0, 2].cpu().numpy(),
                z=self.related_context.y[:, 0].cpu().numpy(),
                name='Prior Points',
                mode='markers', marker=dict(size=2, color='red'),

            ), row=3, col=1)

            # IMPUTED ----
            fig.add_trace(go.Scatter3d(
                x=x_train[:, 0, 1].cpu().numpy(),
                y=x_train[:, 0, 2].cpu().numpy(),
                z=self.imputed_y[:,0].cpu().numpy(),
                name='imputed Prior Points',
                mode='markers', marker=dict(size=2, color='green'),

            ), row=3, col=1)


            # ERROR MODEL -----------------------------------------------------
            y_error = y_train.repeat(1, self.num_related) - self.imputed_y

            error_probs_kernel, kernel_grid, error_logits = self.error_model(
                x_train=x_train[:,:,1:],
                x_test=X_grid[:,:,1:],
                y_error=y_error
            )

            get_lower_upper_surfaces(self.error_model, error_logits, X1, X2, fig, row=2,
                                     col=1, name='error', grid_size=grid_size)

            # ERROR DATA ------
            fig.add_trace(go.Scatter3d(
                x=x_train[:, 0, 1].cpu().numpy(),
                y=x_train[:, 0, 2].cpu().numpy(),
                z=y_error[:,0].cpu().numpy(),
                name='imputed Prior Points',
                mode='markers', marker=dict(size=2, color='red'),

            ), row=2, col=1)


            # PROJECTED PRIOR MODEL ------------------------------------------
            projected_prior_logits, _ = self.error_model.convolve_probs_with_error(
                logits=prior_logits,
                x_train=x_train,
                x_test=X_grid,
                y_error=y_train.repeat(1, self.num_related) - self.imputed_y,
                reverse=False,
            )

            get_lower_upper_surfaces(self.model, projected_prior_logits, X1, X2, fig, row=3, col=2,
                                     grid_size=grid_size,
                                     name='projected_prior')

            # add the scatter plot for x_train, y_train
            fig.add_trace(go.Scatter3d(
                x=self.related_context.x[:, 0, 1].cpu().numpy(),
                y=self.related_context.x[:, 0, 2].cpu().numpy(),
                z=self.related_context.y[:, 0].cpu().numpy(),
                name='Prior Points',
                mode='markers', marker=dict(size=2, color='red'),
            ), row=3, col=2)

            # FINAL PREDICTIONS ------------------------------------------

            mixed_logits = (self.predictions * self.weights.unsqueeze(-1)).sum(dim=1)
            median_predictions = self.model.criterion.median(mixed_logits)

            fig.add_trace(go.Scatter3d(
                x=x_test[:, 0, 1].cpu().numpy(),
                y=x_test[:, 0, 2].cpu().numpy(),
                z=median_predictions.cpu().numpy(),
                name='Final predictions',
                mode='markers', marker=dict(size=2, color='blue'),
            ), row=1, col=1)

            fig.add_trace(go.Scatter3d(
                x=x_test[:, 0, 1].cpu().numpy(),
                y=x_test[:, 0, 2].cpu().numpy(),
                z=median_predictions.cpu().numpy(),
                name='Final predictions',
                mode='markers', marker=dict(size=2, color='blue'),
            ), row=1, col=2)

            predictions = torch.cat([target_logits, projected_prior_logits], dim=1)
            predictions = (predictions * self.weights.unsqueeze(-1)).sum(dim=1)
            get_lower_upper_surfaces(self.model, predictions, X1, X2, fig, row=1, col=2,
                                     name='final_predictions', grid_size=grid_size)

            # TARGET DATA ------
            fig.add_trace(go.Scatter3d(
                x=x_train[:, 0, 1].cpu().numpy(),
                y=x_train[:, 0, 2].cpu().numpy(),
                z=y_train[:, 0].cpu().numpy(),
                mode='markers', marker=dict(size=2, color='red'),
                name='Training Points',
            ), row=1, col=2)

            # PLOT SETTINGS -----------------------------------------------------
            axis = dict(xaxis_title='fidelity', yaxis_title='lambda', zaxis_title='f(x,lambda)')
            fig.update_layout(
                height=2000, width=2000, title_text="Plotly Subplots Example",

                **{f'scene{i}': axis for i in range(1, 6)}

            )

            # fig.show()
            plot(fig)

            # todo counterfactuals
            # counterfactuals = self.error_model.dirac
