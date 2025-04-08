import torch
import numpy as np
import ifbo
from ifbo import Curve, PredictionResult
from ifbo.priors.ftpfn_prior import DatasetPrior

import matplotlib.pyplot as plt
import logging

from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch

logger = logging.getLogger(__name__)

N_LC_PARAMETERS = 23  # Number of parameters for the learning curve basis and their weights


def main():
    # TODO collect a benchmark and sample context and query from it with varying lengths. ----------

    # TODO Collect multiple related datasets from the same benchmark at once -----------------------

    #  MAIN PIPELINE STRUCTURE PROPOSAL ---------------------------------------
    # TODO IFBO anytime performance should be done with neps pipeline but different optimizer in
    #  separate file

    # TODO logging:
    # Dataset fold, nll / calibration, reliability scores, repeated samples over budget
    # allcoations for a fixed total budget -- then stratified over total budgets.

    # (0) Seeding

    # (1) instantiate pfn model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ifbo.surrogate.FTPFN(version="0.0.1", device=device)

    # (2) instantiate meta-train meta-test dataset from benchmark (doing a round-robin?)
    # TODO refactor this into a replacable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=3, single_eval_pos=500)
    context = detokenize_batch(batch)


    # (2.1) sample the configurations & budgets

    # (3) reliability evaluation (at every step?)

    # (4) Collect the experience into context

    # (5) evaluate with the FT-PFN  -- collect for different dataset sizes


if __name__ == '__main__':
    main()
