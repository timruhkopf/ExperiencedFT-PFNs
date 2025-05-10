import math
from copy import copy
from itertools import product
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import logging
import warnings

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import ifbo
from ifbo import Curve, PredictionResult
from ifbo.priors.ftpfn_prior import DatasetPrior
from ifbo.utils import detokenize
from ifbo import Batch

from ifbo.transformer import TransformerModel

from src.evaluation.meta_train_test_split import k_folds, folds_of_size
from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.model.batch_padded_pfn import parse_batch_for_padded_train_data
from src.utils.filelogger import BufferedFileLogger
from src.utils.seeding import SeededRandomContext

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="base_ft", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f'Sweep dir: {Path.cwd()}')
    logger.info(f"Running with config: \n {OmegaConf.to_yaml(cfg, resolve=True)}")
    # fixme: device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device is None \
        else torch.device(cfg.device)
    file_logger = BufferedFileLogger(
        file_name='results.csv',
        file_path='.',
        buffer_size=100,
        header=["metric", "value", 'global_step', "context_size", "task"],
        postfix=["target_task", "train_ids", "seed"]
    )

    # setting up the reference model:
    ftpfn = ifbo.surrogate.FTPFN(
        target_path=Path(cfg.repo_root) / 'data' / '.model',
        version="0.0.1",
        device=device
    )
    pfn_backend: TransformerModel = ftpfn.model
    criterion = pfn_backend.criterion.to(device)

    benchmark = hydra.utils.instantiate(cfg.benchmark.cls, device=device)

    # Generate and select the folds (meta-train-test splits) -------------------
    all_train_ids, test_ids = train_test_split(
        list(range(len(benchmark))),
        test_size=cfg.test_size,
        random_state=cfg.split_seed,  # train test split seed
        shuffle=True
    )

    # allow ourself to rerun certain experiments with specific folds
    folds: List[List[int]] = folds_of_size(all_train_ids, size=cfg.fold_size, drop=True)
    #    folds: List[List[int]] = k_folds(train_ids, k=cfg.k_folds)
    if "fold" in cfg.keys():
        folds = [folds[cfg.fold]]

    if "target_idx" in cfg.keys():
        test_ids = [test_ids[cfg.target_idx]]

    # select the target task and the split of context tasks
    allocation_seeds = range(*cfg.allocation_seeds)
    for train_ids, target_task, seed in tqdm(
            product(folds, test_ids, allocation_seeds),
            total=len(test_ids) * len(folds) * len(allocation_seeds)
    ):
        logger.info(f"Running task: target_task={target_task}, train_ids={train_ids}, seed={seed}")

        file_logger.postfix = [target_task, train_ids, seed]

        # "instantiate" the task and related task datasets (with no budget allocation yet)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            benchmark.collect_task_split(target_id=target_task, train_ids=train_ids)

        # for seed in cfg.allocation_seeds:
        with (SeededRandomContext(seed) as ctx):
            # i, j because we want to make sure, that every task fold sample combination
            # has its own unique budget allocation

            # Allocate budgets on the benchmarks --------------------------
            # sample over multiple meta task sizes and dirichlet alphas
            # TODO for efficiency, we could also repeat the data sampling and then do all of the
            #  tasks at once

            config = dict(
                single_eval_pos=[
                    500,  # target task length
                    *np.random.randint(200, 500, len(train_ids)).tolist()
                ],
                alphas=[10 ** np.random.uniform(-4, -1) for _ in range(len(train_ids) + 1)],
                **cfg.benchmark.sample_config if hasattr(cfg.benchmark, 'sample_config') else {}
            )
            # sample the dirichlet distributed data
            batch = benchmark.sample_batch(**config)

            # parse the batch ---------------------------------------------
            padded_batch = parse_batch_for_padded_train_data(batch, target_idx=0)

            related_task_data = padded_batch.related_tasks
            task_data = padded_batch.target_task

            logger.info(f'Amount of related task data: {related_task_data.observed}')

            target_task_context_x = task_data.x
            target_task_context_y = task_data.y
            target_task_query_x = task_data.query_x
            target_task_query_y = task_data.query_y
            padding_mask = related_task_data.padding_mask
            n_related_tasks = related_task_data.x.shape[1]

            # create and train the model -------------------------------
            model = hydra.utils.instantiate(
                cfg.model.cls,
                model=pfn_backend,
                device=device,
                criterion=criterion,
                logger=file_logger,
                related_task_data=related_task_data,
            )

            train_config = dict(
                query_task_x=target_task_query_x,
                query_task_y=target_task_query_y
            )
            if 'train_call' in cfg.model.keys():
                train_config.update(cfg.model.train_call)

            # distillation will return a prefix, pfn_mixture won't
            model.train(**train_config)

            # Evaluate the model ----------------------------------------------
            evaluator = TestOnNewTaskNLL(
                criterion=pfn_backend.criterion,
                logger=file_logger, device=device
            )

            context_sizes = cfg.context_sizes
            context_sizes = [math.floor(i * target_task_context_x.shape[0]) for i in
                             context_sizes]

            kwargs = {}

            if 'inference_kwargs' in cfg.model.keys():
                kwargs.update(cfg.model.inference_kwargs)

            n_related = related_task_data.x.shape[1]
            evaluator.test_on_new_task(
                model=model,
                context_task_x=target_task_context_x,
                context_task_y=target_task_context_y,
                query_task_x=target_task_query_x,
                query_task_y=target_task_query_y,
                task_name=cfg.model.meta.name,
                step=0,
                context_sizes=context_sizes,
                # fwd kwargs
                **kwargs
            )

            # Quick Baselines ----------------------------------------------
            # sanity check: what if we put in the current task as context.
            # evaluator.test_on_new_task(
            #     model=pfn_backend,
            #     task_name=f'Baseline: complete target_task_x as context',
            #     context_task_x=target_task_context_x,
            #     context_task_y=target_task_context_y,
            #     query_task_x=target_task_query_x,
            #     query_task_y=target_task_query_y
            # )

            # Sanity check: what if we took the complete context from the related task and attempted
            # to predict the current task
            evaluator.test_on_new_task(
                model=pfn_backend,
                task_name=f'Baseline: approx. conditioning on complete related',
                prefix_x=related_task_data.x[:-25],
                prefix_y=related_task_data.y[:-25],
                context_task_x=target_task_context_x.repeat(1, n_related_tasks, 1),
                context_task_y=target_task_context_y.repeat(1, n_related_tasks),
                query_task_x=target_task_query_x.repeat(1, n_related_tasks, 1),
                query_task_y=target_task_query_y.repeat(1, n_related_tasks),
                context_sizes=context_sizes,
                src_key_padding_mask=padding_mask
            )

            evaluator.test_on_new_task(
                model=pfn_backend,
                task_name='Baseline: naked pfn',
                context_task_x=target_task_context_x,
                context_task_y=target_task_context_y,
                query_task_x=target_task_query_x,
                query_task_y=target_task_query_y,
                context_sizes=context_sizes
            )

            if cfg.model.meta.name == 'distill':
                # Sanity check: the initial points of optimization added as context
                evaluator.test_on_new_task(
                    model=pfn_backend,
                    task_name='x_init context (no-distillation)',
                    prefix_x=model._init_x,
                    prefix_y=model._init_y,
                    context_task_x=target_task_context_x.repeat(1, n_related_tasks, 1),
                    context_task_y=target_task_context_y.repeat(1, n_related_tasks),
                    query_task_x=target_task_query_x.repeat(1, n_related_tasks, 1),
                    query_task_y=target_task_query_y.repeat(1, n_related_tasks),
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


if __name__ == '__main__':
    from src.utils.resolvers import *

    main()
