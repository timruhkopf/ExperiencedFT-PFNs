import abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class AbstractCallback(abc.ABC):
    __name__ = 'AbstractCallback'
    DEBUG = False
    STOP_AT = -1

    def __init__(self, model, error_model, related_context, logger, device, **kwargs):
        self.logger = logger
        self.device = device
        self.model = model
        self.error_model = error_model
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
            target_logits, prior_logits, error_logits, imputed_y, y_error,
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
