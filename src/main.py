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
    # TODO collect a benchmark and sample context and query from it with varying lengths. ----------

    # TODO Collect multiple related datasets from the same benchmark at once -----------------------

    # DEMO on how to sample the Prior and evaluate curves off of it.  ------------------------------
    n_hyperparameter_dims = 3
    dataset = DatasetPrior(num_params=n_hyperparameter_dims, num_outputs=N_LC_PARAMETERS)
    n_hyperparameters = 2
    configs = np.random.uniform(
        size=(n_hyperparameters, 3))  # 5 configurations with 3 parameters each

    curve_func = dataset.curves_for_configs(configs)
    # @Tim Notice, how the curve function here allows to evaluate the entire basket
    # at any point of interest.
    x = np.linspace(0, 1, 100)
    # Evaluate for the second configuration

    curve_func(x, 0)  # Evaluate the curve for the second configuration


    #  MAIN PIPELINE STRUCTURE PROPOSAL ---------------------------------------
    # TODO IFBO anytime performance should be done with neps pipeline but different optimizer in
    # separate file

    # TODO logging:
    # Dataset fold, nll / calibration, reliability scores, repeated samples over budget
    # allcoations for a fixed total budget -- then stratified over total budgets.

    # (0) Seeding

    # (1) instantiate pfn model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ifbo.surrogate.FTPFN(version="0.0.1", device=device)

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)

    # (2.1) sample the configurations & budgets

    # (3) reliability evaluation (at every step?)

    # (4) Collect the experience into context

    # (5) evaluate with the FT-PFN  -- collect for different dataset sizes







if __name__ == '__main__':
    main()
