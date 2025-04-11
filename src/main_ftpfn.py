from copy import copy
from typing import List

import torch
import numpy as np
import ifbo
from ifbo import Curve, PredictionResult, BarDistribution
from ifbo.priors.ftpfn_prior import DatasetPrior

import matplotlib.pyplot as plt
import logging

from ifbo.transformer import TransformerModel
from ifbo.utils import detokenize, tokenize

from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch
from ifbo import Batch

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
    # batched evaluation -----------------------
    from src.dataset.taskprior import MetaTaskPriorSameProblem
    from ifbo.transformer import TransformerModel

    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=5, single_eval_pos=500)
    pfn_backend: TransformerModel = model.model

    single_eval_pos = batch.single_eval_pos[0]
    cx, qx, cy, qy = batch.x[:single_eval_pos], batch.x[:single_eval_pos], \
        batch.y[:single_eval_pos, :], batch.y[:single_eval_pos, :]

    logits = pfn_backend.forward((torch.cat([cx, qx], dim=0), cy), single_eval_pos=cx.shape[0])
    criterion: BarDistribution = pfn_backend.criterion

    # quantile
    q = 0.05
    lower_bound = criterion.icdf(logits, q)
    upper_bound = criterion.icdf(logits, 1 - q)
    median = criterion.icdf(logits, 0.5)


    # (3) reliability evaluation (at every step?)
    nll = -criterion(logits, qy)
    # FIXME: each task will need a separate fwd, because of
    # single_eval_pos that may differ, which does not work with batch, but we can check for
    # different levels of the target task at varying lengths of the context


    # (4) Collect the experience into context

    # (5) evaluate with the FT-PFN  -- collect for different dataset sizes


if __name__ == '__main__':
    main()
