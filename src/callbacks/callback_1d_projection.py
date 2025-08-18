import torch
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

from callbacks.abstract_callback import AbstractCallback


def get_lower_upper_surfaces(criterion, logits, x1, x2, fig, row, col, grid_size,
                             name='', ):
    lower = criterion.icdf(logits, 0.25).numpy().reshape(-1)
    mean = criterion.mean(logits).numpy().reshape(-1)
    upper = criterion.icdf(logits, 0.75).numpy().reshape(-1)

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
        opacity=0.7, showscale=False, coloraxis=None
        #             name=f'75th Q {name}'
    ),
        row=row, col=col
    )


class Callback1dProjectionPFNContext(AbstractCallback):
    __name__ = 'Callback1DProjectionPFNContext'
    STOP_AT = 301

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, projected_logits, convolved_criterion,
            imputed_y, y_error,
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
        print()
        if self.step(x_train) == self.STOP_AT:
            # make a meshgrid for the x_grid
            grid_size = 50
            x1 = np.linspace(0, 1, grid_size)
            x2 = np.linspace(0, 1, grid_size)
            X1, X2 = np.meshgrid(x1, x2)
            config_id = torch.tensor((X1 * 999).ravel()).floor()
            X_grid = torch.vstack([config_id, torch.tensor(X1.ravel()), torch.tensor(X2.ravel())]).T
            X_grid = X_grid.to(self.device).float().unsqueeze(1)

            fig = make_subplots(
                rows=2, cols=self.num_related + 1,
                subplot_titles=["Target",  *['' for _ in range(self.num_related)], 'merged', *[
                    f"Prior {b}"  for b in range( self.num_related)]] ,
                specs=[[{"type": "surface"}] + [{"type": "scatter"}] +
                       [{"type": "surface"} for _ in range(self.num_related -1)],
                       [{"type": "surface"}] + [{"type": "surface"} for _ in range(self.num_related)]]
            )

            # WEIGHTS -----------------------------------------------------
            df = self.logger.df[self.logger.df['metrics'] == 'weights']

            pattern = r'^weight_\d+$'

            for col in df.columns[df.columns.to_series().str.match(pattern)]:
                fig.add_trace(go.Scatter(x=df['step'], y=df[col], mode='lines', name=col), row=1,
                              col=2)

            # FT-PFN (Meta-UN-aware) -----------------------------------------------------
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

            get_lower_upper_surfaces(
                self.model.criterion, target_logits, X1, X2, fig,
                row=1, col=1,
                name='target', grid_size=grid_size
            )

            # META-AWARE (joint) -----------------------------------------------------
            target_surface_logits = self.parent_model.strategy(x_train, X_grid, y_train, inc)
            get_lower_upper_surfaces(
                self.model.criterion, target_surface_logits, X1, X2, fig,
                row=2, col=1,
                name='meta-aware', grid_size=grid_size
            )

            # META-AWARE (Marginals) -----------------------------------------------------
            logits = self.parent_model.interim_results['logits']
            additional_target_locations = []
            if logits.shape[1] > 1:
                additional_target_locations = [(2, b + 2) for b in range(self.num_related)]
                logits = logits[:,1:, :]
                for b in range(self.num_related):
                    prior_col = b + 2
                    get_lower_upper_surfaces(
                        self.model.criterion, logits[:, b, :], X1, X2, fig,
                        row=2, col=prior_col, name=f'prior {b}', grid_size=grid_size
                    )

                # PRIOR DATA -----------------------------------------------------
                for b in range(self.num_related):
                    prior_col = b + 2
                    # add the scatter plot for x_train, y_train
                    fig.add_trace(go.Scatter3d(
                        x=self.related_context.x[:, b, 1].cpu().numpy(),
                        y=self.related_context.x[:, b, 2].cpu().numpy(),
                        z=self.related_context.y[:, b].cpu().numpy(),
                        name='Prior Points',
                        mode='markers', marker=dict(size=2, color='red'),
                    ), row=2, col=prior_col)


            # TARGET DATA ------
            target_locations =  [(1, 1), (2, 1), *additional_target_locations]
            for (row, col) in target_locations:
                # TARGET DATA ------
                # add the scatter plot for x_train, y_train
                fig.add_trace(go.Scatter3d(
                    x=x_train[:, 0, 1].cpu().numpy(),
                    y=x_train[:, 0, 2].cpu().numpy(),
                    z=y_train[:, 0].cpu().numpy(),
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=x_train[:, 0, 0].cpu().numpy(),  # Color by step
                        colorscale='Viridis',  # Or any colorscale you like
                        colorbar=dict(title='Step')
                    ),
                    name='Training Points',
                ), row=row, col=col)

            fig.update_layout(
                height=1000, width=2000, title_text="Plotly Subplots Example",
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


class Callback1dProjectionPI(AbstractCallback):
    __name__ = 'Callback1DProjectionPI'
    STOP_AT = 11

    def on_trained_ppds(
            self,
            target_logits, prior_logits, error_logits, projected_logits, convolved_criterion,
            imputed_y, y_error,
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
        print()
        if self.step(x_train) == self.STOP_AT:


            self.error_model = self.parent_model.strategy.error_model

            B = self.num_related
            rows, cols = 3, B + 1  # 3 rows, 2 + (B-1) columns
            precision = 2
            formatted_weights = [f"{w:.{precision}f}" for w in self.weights.tolist()]
            titles = np.array([
                ["Target"] + [f"Error {b}" for b in range(B)],

                [""] + [f"Prior {b}" for b in range(B)],

                [f"Final Prediction with weights {formatted_weights}"] + \
                [f"Projected Prior {b}" for b in range(B)]
            ])

            specs = [[{"type": "surface"} for _ in range(cols)] for _ in range(rows)]
            specs[1][0] = {"type": "scatter"}  # Target data

            # Create subplot figure with 2 rows and 2 columns
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=titles.flatten().tolist(),
                specs=specs,
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

            get_lower_upper_surfaces(
                self.model.criterion, target_logits, X1, X2, fig,
                row=1, col=1,
                name='target', grid_size=grid_size
            )

            target_locations = {(row, col) for row in range(2, rows + 1)
                                for col in range(1, cols + 1)}

            target_locations.add((1, 1))
            target_locations.discard((2, 1))

            for (row, col) in target_locations:
                # TARGET DATA ------
                # add the scatter plot for x_train, y_train
                fig.add_trace(go.Scatter3d(
                    x=x_train[:, 0, 1].cpu().numpy(),
                    y=x_train[:, 0, 2].cpu().numpy(),
                    z=y_train[:, 0].cpu().numpy(),
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=x_train[:, 0, 0].cpu().numpy(),  # Color by step
                        colorscale='Viridis',  # Or any colorscale you like
                        colorbar=dict(title='Step')
                    ),
                    # marker=dict(size=2, color='cyan'),
                    name='Training Points',
                ), row=row, col=col)

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

            for b in range(self.num_related):
                prior_col = b + 2
                get_lower_upper_surfaces(
                    self.model.criterion, prior_logits[:, b, :], X1, X2, fig,
                    row=2, col=prior_col, name='prior', grid_size=grid_size
                )

                # PRIOR DATA ------
                # add the scatter plot for x_train, y_train
                fig.add_trace(go.Scatter3d(
                    x=self.related_context.x[:, b, 1].cpu().numpy(),
                    y=self.related_context.x[:, b, 2].cpu().numpy(),
                    z=self.related_context.y[:, b].cpu().numpy(),
                    name='Prior Points',
                    mode='markers', marker=dict(size=2, color='red'),
                ), row=2, col=prior_col)

                # IMPUTED ----
                fig.add_trace(go.Scatter3d(
                    x=x_train[:, 0, 1].cpu().numpy(),
                    y=x_train[:, 0, 2].cpu().numpy(),
                    z=self.imputed_y[:, b].cpu().numpy(),
                    name='imputed Prior Points',
                    mode='markers', marker=dict(size=2, color='purple'),
                ), row=2, col=prior_col)


            # ERROR MODEL -----------------------------------------------------
            y_error = y_train.repeat(1, self.num_related) - self.imputed_y
            # FIXME: check this in the original code

            error_probs_kernel, kernel_grid, error_logits = self.error_model(
                x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
                x_test=X_grid[:, :, 1:].repeat(1, self.num_related, 1),
                y_error=y_error
            )
            for b in range(self.num_related):
                error_col = b + 2
                get_lower_upper_surfaces(
                    self.error_model.criterion, error_logits[:, b, :],
                    X1, X2, fig,
                    row=1, col=error_col, name='error', grid_size=grid_size
                )

                # ERROR DATA ------
                fig.add_trace(go.Scatter3d(
                    x=x_train[:, 0, 1].cpu().numpy(),
                    y=x_train[:, 0, 2].cpu().numpy(),
                    z=y_error[:, b:b + 1].flatten().cpu().numpy(),
                    name='imputed Prior Points',
                    mode='markers', marker=dict(size=2, color='yellow'),
                ), row=1, col=error_col)

            # PROJECTED PRIOR MODEL ------------------------------------------
            projected_prior_logits_grid, _, conv_criterion = (
                self.error_model.convolve_probs_with_error(
                    logits=prior_logits,
                    logits_borders=self.model.criterion.borders,
                    x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
                    # TODO check this in the original code
                    x_test=X_grid[:, :, 1:].repeat(1, self.num_related, 1),
                    y_error=y_error,
                    reverse=False,
                ))

            # PROJECTED PRIOR DATA ------
            prior_logits = self.model(
                (
                    torch.cat([
                        self.related_context.x,
                        self.related_context.x
                    ]),
                    self.related_context.y
                ),
                single_eval_pos=self.related_context.x.shape[0]
            )

            projected_prior_logits, _, conv_criterion = self.error_model.convolve_probs_with_error(
                logits=prior_logits,
                logits_borders=self.model.criterion.borders,
                x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
                # TODO check this in the original code
                x_test=self.related_context.x[:, :, 1:],  # TODO check this in the original code
                y_error=y_error,
                reverse=False,
            )

            for b in range(self.num_related):
                projected_col = b + 2

                get_lower_upper_surfaces(
                    conv_criterion, projected_prior_logits_grid[:, b:b + 1, :],
                    X1, X2, fig,
                    row=3, col=projected_col,
                    grid_size=grid_size,
                    name='projected_prior')

                # TRUE PRIOR DATA ------
                # add the scatter plot for x_train, y_train
                fig.add_trace(go.Scatter3d(
                    x=self.related_context.x[:, b, 1].cpu().numpy(),
                    y=self.related_context.x[:, b, 2].cpu().numpy(),
                    z=self.related_context.y[:, b].cpu().numpy(),
                    name='Prior Points',
                    mode='markers', marker=dict(size=2, color='red'),
                ), row=3, col=projected_col)

                projected_prior_points = conv_criterion.median(projected_prior_logits[:, b, :])

                fig.add_trace(go.Scatter3d(
                    x=self.related_context.x[:, b, 1].cpu().numpy(),
                    y=self.related_context.x[:, b, 2].cpu().numpy(),
                    z=projected_prior_points.cpu().numpy(),
                    name='Projected Prior Points',
                    mode='markers', marker=dict(size=2, color='yellow'),
                ), row=3, col=projected_col)

                # DIRAC PROJECTED PRIOR DATA ------
                dirac_prior_logits, _,  bardist = self.error_model.dirac_forward(
                    x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
                    dirac_x=self.related_context.x[:, :, 1:],
                    dirac_y=self.related_context.y,
                    y_error=y_error,
                    reverse=False
                )

                dirac_prior_points = bardist.median(dirac_prior_logits)
                fig.add_trace(go.Scatter3d(
                    x=self.related_context.x[:, b, 1].cpu().numpy(),
                    y=self.related_context.x[:, b, 2].cpu().numpy(),
                    z=dirac_prior_points[:, b].cpu().numpy(),
                    name='Dirac Prior Points',
                    mode='markers', marker=dict(size=2, color='orange'),
                ), row=3, col=projected_col)

            # FINAL PREDICTIONS ------------------------------------------

            mixed_logits = (self.predictions * self.weights.unsqueeze(-1)).sum(dim=1)
            median_predictions = self.model.criterion.median(mixed_logits)

            # FINAL PREDICTIONS DATA - target task plot ----
            fig.add_trace(go.Scatter3d(
                x=x_test[:, 0, 1].cpu().numpy(),
                y=x_test[:, 0, 2].cpu().numpy(),
                z=median_predictions.cpu().numpy(),
                name='Final predictions',
                mode='markers', marker=dict(size=2, color='blue'),
            ), row=3, col=1)

            # FINAL PREDICTIONS DATA - mixed plot ----
            fig.add_trace(go.Scatter3d(
                x=x_test[:, 0, 1].cpu().numpy(),
                y=x_test[:, 0, 2].cpu().numpy(),
                z=median_predictions.cpu().numpy(),
                name='Final predictions',
                mode='markers', marker=dict(size=2, color='blue'),
            ), row=3, col=1)

            projected_logits_grid = projected_prior_logits_grid
            lower = torch.where(conv_criterion.borders[conv_criterion.borders >= 0].min(
            ) == conv_criterion.borders)[0].item()
            upper = torch.where(conv_criterion.borders[conv_criterion.borders <= 1].max(
            ) == conv_criterion.borders)[0].item()
            projected_logits_grid = projected_logits_grid[:, :, lower:upper + 1]

            predictions = torch.cat([target_logits, projected_logits_grid], dim=1)
            predictions = (predictions * self.weights.unsqueeze(-1)).sum(dim=1)
            get_lower_upper_surfaces(
                self.model.criterion, predictions, X1, X2, fig,
                row=3, col=1,
                name='final_predictions', grid_size=grid_size
            )

            # Counterfactuals -----------------------------------------------------
            # step = x_train.shape[0]
            #
            # counterfactural_logits, error_logits, bardist = self.error_model.dirac_forward(
            #     x_train=x_train[:, :, 1:].repeat(1, self.num_related, 1),
            #     dirac_x=self.related_context.x[:, :, 1:],
            #     dirac_y=self.related_context.y,
            #     y_error=y_error,
            #     reverse=False
            # )
            #
            # # Now we collect the y values for the counterfactual data.
            #
            # counterfactual_y = torch.stack([
            #     bardist.median(counterfactural_logits[:, b, :].squeeze(1))
            #     for b in range(self.num_related)
            # ], dim=1).to(self.device)
            #
            # related_x = self.related_context.x
            # query = x_train.repeat(1, self.num_related, 1)
            #
            #
            # # Get the logits for the target task data under the counterfactual prior PPD
            # prior_counterfactual_logits = self.model(
            #     (
            #         torch.cat(
            #             [
            #                 related_x,
            #                 query
            #             ]
            #         ),
            #         counterfactual_y
            #     ),
            #     single_eval_pos=self.related_context.x.shape[0],
            # )
            #
            # # collect median counterfactual predictions
            # for b in range(self.num_related):
            #     median_counterfactual_predictions = self.model.criterion.median(
            #         prior_counterfactual_logits[:, b, :]
            #     )
            #
            #     prior_col = b + 2
            #
            #     # add the scatter plot for x_train, y_train
            #     fig.add_trace(go.Scatter3d(
            #     x=x_train[:, 0, 1].cpu().numpy(),
            #     y=x_train[:, 0, 2].cpu().numpy(),
            #     z=median_counterfactual_predictions.cpu().numpy(),
            #     name='Counterfactual Predictions',
            #     mode='markers', marker=dict(size=2, color='cyan'),
            # ), row=3, col= prior_col)

            # WEIGHTS -----------------------------------------------------
            df = self.logger.df[self.logger.df['metrics'] == 'weights']

            pattern = r'^weight_\d+$'

            for col in df.columns[df.columns.to_series().str.match(pattern)]:
                fig.add_trace(go.Scatter(x=df['step'], y=df[col], mode='lines', name=col), row=2, col=1)

            # PLOT SETTINGS -----------------------------------------------------
            axis = dict(xaxis_title='fidelity', yaxis_title='lambda', zaxis_title='f(x,lambda)')
            fig.update_layout(
                height=2000, width=2000, title_text="Plotly Subplots Example",

                **{f'scene{i}': axis for i in range(1, 6)}

            )

            # fig.show()
            plot(fig)
            plt.clf()
