import torch
import numpy as np
import ifbo
from ifbo import Curve, PredictionResult
from ifbo.priors.ftpfn_prior import DatasetPrior

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


N_LC_PARAMETERS = 23  # Number of parameters for the learning curve basis and their weights

def main():

    # DEMO on how to sample the Prior and evaluate curves off of it.  ------------------------------
    n_hyperparameter_dims = 3
    dataset = DatasetPrior(num_params=n_hyperparameter_dims, num_outputs=N_LC_PARAMETERS)
    n_hyperparameters = 2
    configs = np.random.uniform(size=(n_hyperparameters, 3))  # 5 configurations with 3 parameters each

    curve_func = dataset.curves_for_configs(configs)
    # @Tim Notice, how the curve function here allows to evaluate the entire basket
    # at any point of interest.
    x = np.linspace(0, 1, 100)
 # Evaluate for the second configuration

    if False:

        for i, config in enumerate(configs):
            y= curve_func(x, i)
            plt.plot(x, y, label=f'Config {i+1}', alpha=0.5)

        plt.xlabel('Budget')
        plt.ylabel('Performance')
        plt.title('Learning Curves for Configurations')
        plt.legend()
        plt.show()


    # Instantiate the FT-PFN model and fwd with it -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ifbo.surrogate.FTPFN(version="0.0.1", device=device)

    context = [
        Curve(hyperparameters=torch.tensor([0.2, 0.1, 0.5]), t=torch.tensor([0.1, 0.2, 0.3]),
              y=torch.tensor([0.1, 0.15, 0.3])),
        Curve(hyperparameters=torch.tensor([0.2, 0.3, 0.25]), t=torch.tensor([0.1, 0.2, 0.3, 0.4]),
              y=torch.tensor([0.2, 0.5, 0.6, 0.75])),
    ]

    query = [ # notice, that Curve is used, but y is not provided
        Curve(hyperparameters=torch.tensor([0.2, 0.1, 0.5]),
              t=torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1])),
        Curve(hyperparameters=torch.tensor([0.2, 0.3, 0.25]),
              t=torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])),
        Curve(hyperparameters=torch.tensor([0.8, 0.2, 0.6]),
              t=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])),
    ]

    predictions: list[PredictionResult] = model.predict(context=context, query=query)

    if False:
        for curve in context:
            plt.plot(curve.t, curve.y, color=curve.hyperparameters.numpy().tolist() + [0.5])

        for i, (curve, pred) in enumerate(zip(query, predictions)):
            c = curve.hyperparameters.numpy().tolist() + [0.2]
            y025, y975 = pred.quantile(0.025).cpu(), pred.quantile(0.975).cpu()
            label = f"Curve {i + 1}: " + str(
                [round(_, 1) for _ in curve.hyperparameters.numpy().tolist()])
            torch.round(curve.hyperparameters, decimals=1).tolist()
            plt.fill_between(curve.t, y025, y975, color=c, label=label)

        plt.ylim(0, 1), plt.xlim(None, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="")
        plt.xlabel("step ($t$)")
        plt.ylabel("performance ($y$)")
        plt.title("FT-PFN predictions with 95% confidence interval")
        plt.show()


    # Sample incomplete curves with (context and query split) ---------------------------------

    # @TIM Notice This is the amount of budget spent (epochs) across the curves
    single_eval_pos = 700  # number of observations in the context (< seq_len).

    # TODO: investigate what part is the Dirichlet exactly!
    batch = ifbo.priors.ftpfn_prior.get_batch(
        batch_size=1,
        seq_len=1000,  # maximum number of observations per task for training. Default: 1000.
        num_features=12,
        # sample the dimension of HPs with Uniform(1, num_features-1). Default: 12.
        single_eval_pos=single_eval_pos
    )

    batch.__dict__.keys()
    # all of these are of [batch, seq_len, specific dimensions]
    batch.x.shape
    batch.y.shape
    batch.target_y.shape

    all(batch.target_y == batch.y)

    context, query = ifbo.utils.detokenize(batch, context_size=single_eval_pos, device=device)




if __name__ == '__main__':
    main()