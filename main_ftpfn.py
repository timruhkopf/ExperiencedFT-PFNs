import math
from copy import copy
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

import ifbo
from ifbo import Curve, PredictionResult
from ifbo.priors.ftpfn_prior import DatasetPrior


import logging

from ifbo.utils import detokenize
from ifbo import Batch


from pfns4hpo.transformer import TransformerModel
from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch


from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.utils.filelogger import BufferedFileLogger
from src.utils.parse_batch import parse_batch
from src.utils.seeding import SeededRandomContext

logger = logging.getLogger(__name__)

N_LC_PARAMETERS = 23  # Number of parameters for the learning curve basis and their weights


@hydra.main(config_path="configs", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f'Current working directory: {Path.cwd()}')
    logger.info(f"Running with config: \n {OmegaConf.to_yaml(cfg, resolve=True)}")
    # fixme: device
    device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_logger = BufferedFileLogger(
        file_name='results.csv',
        file_path='.',
        buffer_size=1000,
        header=("metric", "value", "task", "context_size"),
        postfix=[]
    )

    # setting up the reference model:
    ftpfn = ifbo.surrogate.FTPFN(version="0.0.1", device=device)
    pfn_backend: TransformerModel = ftpfn.model
    criterion = pfn_backend.criterion.to(device)

    with SeededRandomContext(cfg.benchmark_seed) as ctx:

        # TODO iterate over the batch samples from the benchmark
        benchmark = hydra.utils.instantiate(cfg.benchmark.cls, device=device)
        batch = benchmark.sample_batch(**cfg.benchmark.sample_config)

        if "n_prefix_tokens" in cfg.model.meta.keys():
            n_prefix_tokens = cfg.model.meta["n_prefix_tokens"]
        else:
            n_prefix_tokens = 0
        data = parse_batch(batch, cfg.target_idx, n_prefix_tokens=n_prefix_tokens)

        target_task_context_x = data['target_task_context']['x']
        target_task_context_y = data['target_task_context']['y']
        target_task_query_x = data['target_task_query']['x']
        target_task_query_y = data['target_task_query']['y']

    with SeededRandomContext(cfg.seed) as ctx:
        model = hydra.utils.instantiate(
            cfg.model.cls,
            device=device,
            criterion=criterion,
            logger=file_logger,
            related_task_data=data['related_task_data'],
        )

        prefix = model.train(benchmark)

        evaluator = TestOnNewTaskNLL(
            criterion=pfn_backend.criterion,
            logger=file_logger, device=device
        )

        context_sizes = cfg.context_sizes
        context_sizes = [math.floor(i * target_task_context_x.shape[0]) for i in context_sizes]

        evaluator.test_on_new_task(
            model=model,
            prefix_x=prefix[0],
            prefix_y=prefix[1],
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            task_name=cfg.model.meta,
            step=0,
            context_sizes=context_sizes,
            # fwd kwargs
            **cfg.model.inference_kwargs
        )

        # Quick Baselines --------------------
        evaluator.test_on_new_task(
            model=pfn_backend,
            task_name='naked pfn',
            # prefix_x=baseline_x,
            # prefix_y=baseline_y,
            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,
            context_sizes=context_sizes
        )

        # Intensely checking sanity baselines: for each task
        # for task in data['related_task_data']:
        #     x_task_context = task['x']
        #     y_task_context = task['y']
        #
        #     # Adding in the entire context of the related task
        #     evaluator.test_on_new_task(
        #         model=pfn_backend,
        #         task_name='baseline (full context task 0)',
        #         prefix_x=x_task_context,
        #         prefix_y=y_task_context,
        #         context_task_x=target_task_context_x,
        #         context_task_y=target_task_context_y,
        #         query_task_x=target_task_query_x,
        #         query_task_y=target_task_query_y,
        #         context_sizes=context_sizes
        #     )
        #
        #     # adding in half of the related dataset in
        #     half_x = x_task_context.shape[0] // 2
        #     evaluator.test_on_new_task(
        #         model=pfn_backend,
        #         task_name='baseline (half context task 0)',
        #         prefix_x=x_task_context[:half_x],
        #         prefix_y=y_task_context[:half_x],
        #         context_task_x=target_task_context_x,
        #         context_task_y=target_task_context_y,
        #         query_task_x=target_task_query_x,
        #         query_task_y=target_task_query_y,
        #         context_sizes=context_sizes
        #     )
        #
        #     # Adding in (almost) the entire context of the related task
        #     # we can't fit the entire one, since we need space in the sequence
        #     # to do a batched evaluation over the query points
        #     evaluator.test_on_new_task(
        #         model=pfn_backend,
        #         task_name='baseline (approx. complete context task 0)',
        #         prefix_x=x_task_context,
        #         prefix_y=y_task_context,
        #         context_task_x=target_task_context_x[:-25],
        #         context_task_y=target_task_context_y[:-25],
        #         query_task_x=target_task_query_x,
        #         query_task_y=target_task_query_y,
        #         context_sizes=context_sizes  # [:10]
        #     )

    file_logger.close()

    return -1


def main_demo():
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
    # TODO refactor this into a replicable dataset class
    prior = MetaTaskPriorSameProblem(dim_hyperparameters=3, n_fidelities=None, seq_len=1000)
    batch = prior.sample_batch(n_tasks=3, single_eval_pos=500)
    context = detokenize_batch(batch)

    target_task = context[0]
    related_tasks = context[1:]

    # (3) reliability evaluation (at every step?)
    # TODO: sanity check: how do i get a curve over context lengths (amount of data from the
    #  target_task --> take the dirichlet ordering (which is causing the token ordering in the
    #  context)
    # nll = []
    # for task in related_tasks:
    #     predictions:List[PredictionResult] = model.predict(context=task, query=target_task)
    #     nll_score = sum([sum(-prediction.likelihood(q.y).cpu())
    #                for prediction, q in zip(predictions, target_task)])
    #     nll.append(nll_score)

    def generate_round_robin_batches(batch, max_steps: int, min_context=10, stepsize=10):
        """
        Generates modified batches with incrementing single_eval_pos for target task
        while maintaining original splits for related tasks.

        Args:
            batch: Contains x, y, single_eval_pos as described
            max_steps: Maximum number of evaluation position increments

        Returns:
            Generator yielding (target_idx, modified_batch) tuples
        """
        x, y, original_sep = batch.x, batch.y, batch.single_eval_pos
        batch_size = batch.x.size(1)

        for target_idx in range(batch_size):

            # Determine maximum possible offset for this sequence
            seq_len = x.size(0)
            original_split = original_sep[target_idx]

            for modified_sep in range(min_context, original_split + 1, stepsize):
                # Create new target_batch with modified split positions
                target_batch = Batch(
                    x=x[:, target_idx:target_idx + 1],
                    y=y[:, target_idx:target_idx + 1],
                    target_y=y[:, target_idx:target_idx + 1],
                    single_eval_pos=modified_sep
                )

                target_context, target_query = detokenize(target_batch, context_size=modified_sep)

                yield target_idx, target_context, target_query, modified_sep

    x, y, original_sep = batch.x, batch.y, batch.single_eval_pos
    batch_size = batch.x.size(1)
    original_sep = copy(original_sep)
    results = {}  # Dict[Tuple[target_idx, related, single_eval_pos], float]
    for target_idx, target_context, _, sep in generate_round_robin_batches(batch, max_steps=5):

        assert sep == sum(len(curve.t) for curve in target_context), \
            f"Expected sep {sep}, but got {sum(len(curve.t) for curve in target_context)}"
        context_tasks = [context[i] for i in range(batch_size) if i != target_idx]

        for r, related_task in enumerate(context_tasks):
            print(target_idx, r, sep)
            # Your existing processing pipeline
            # fixme: using the predict method on each combination is super inefficient.
            #  quicker way: collect the context and query tokens and use the fwd directly.
            #  but this will require us to change the way that we need to calcualte the nll as well
            predictions = model.predict(related_task, target_context)

            nll_score = torch.cat([(-prediction.likelihood(q.y).cpu())
                                   for prediction, q in zip(predictions, target_context)])

            results[(target_idx, r, sep)] = nll_score

    # for key, value in results.items():
    #     results[key] = value.item()

    processed_results = [(*k, v.quantile(q=0.5).item(), v.quantile(q=0.95).item(), v.quantile(
        q=0.05).item()) for k, v in results.items()]

    # get the pd.DataFrame from the results
    import pandas as pd
    results_df = pd.DataFrame(processed_results,
                              columns=['Task', 'Related', 'SingleEvalPos', 'median NLL',
                                       'upper NLL', 'lower NLL']
                              )
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Ensure proper sorting for line connections
    results_df = results_df.sort_values(['Task', 'Related', 'SingleEvalPos'])

    # Create FacetGrid with error bands
    g = sns.FacetGrid(results_df, col='Task', col_wrap=3, height=4,
                      sharey=False, despine=False)
    g.map_dataframe(sns.lineplot, x='SingleEvalPos', y='median NLL',
                    hue='Related', style='Related',
                    palette='viridis', markers=True, dashes=False,
                    errorbar=None)  # We'll add CI manually

    # Add error bands using fill_between
    for ax, task in zip(g.axes.flat, results_df['Task'].unique()):
        task_data = results_df[results_df['Task'] == task]
        for rel, color in zip(task_data['Related'].unique(), sns.color_palette('viridis')):
            rel_data = task_data[task_data['Related'] == rel]
            ax.fill_between(rel_data['SingleEvalPos'],
                            rel_data['lower NLL'],
                            rel_data['upper NLL'],
                            color=color, alpha=0.2)

    g.add_legend(title='Related')
    g.set_axis_labels('SingleEvalPos', 'Median NLL with CI')
    g.set_titles("Task {col_name}")
    plt.tight_layout()
    plt.show()

    # (4) Collect the experience into context

    # (5) evaluate with the FT-PFN  -- collect for different dataset sizes


if __name__ == '__main__':
    from src.utils.resolvers import *

    main()
