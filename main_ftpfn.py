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

from ifbo.transformer import TransformerModel
from src.dataset.taskprior import MetaTaskPriorSameProblem, detokenize_batch

from src.evaluation.test_on_new_task_nll import TestOnNewTaskNLL
from src.model.batch_padded_pfn import parse_batch_for_padded_train_data
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
    device = torch.device('cpu')  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_logger = BufferedFileLogger(
        file_name='results.csv',
        file_path='.',
        buffer_size=1000,
        header=("metric", "value", 'global_step', "context_size", "task"),
        postfix=[]
    )

    # setting up the reference model:
    ftpfn = ifbo.surrogate.FTPFN(
        target_path=Path(cfg.repo_root) / 'data' / '.model',
        version="0.0.1",
        device=device
    )
    pfn_backend: TransformerModel = ftpfn.model
    criterion = pfn_backend.criterion.to(device)

    with (SeededRandomContext(cfg.benchmark_seed) as ctx):

        # TODO train test split on tasks, pass to the benchmark instance

        # TODO iterate over the batch samples from the benchmark
        benchmark = hydra.utils.instantiate(cfg.benchmark.cls, device=device)

        config = {}
        if 'sample_config' in cfg.benchmark.keys():
            config.update(cfg.benchmark.sample_config)
        batch = benchmark.sample_batch(**config)

        if "n_prefix_tokens" in cfg.model.meta.keys():
            n_prefix_tokens = cfg.model.meta["n_prefix_tokens"]
        else:
            n_prefix_tokens = 0
        # data = parse_batch(batch, cfg.target_idx, n_prefix_tokens=n_prefix_tokens)

        padded_batch = parse_batch_for_padded_train_data(batch, target_idx=cfg.target_idx)

        related_task_data = padded_batch.related_tasks
        task_data = padded_batch.target_task

        logger.info(f'Amount of related task data: {related_task_data.observed}')

        target_task_context_x = task_data.x
        target_task_context_y = task_data.y
        target_task_query_x = task_data.query_x
        target_task_query_y = task_data.query_y
        padding_mask = related_task_data.padding_mask
        n_related_tasks = related_task_data.x.shape[1]

    with SeededRandomContext(cfg.seed) as ctx:
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

        prefix = model.train(**train_config)

        evaluator = TestOnNewTaskNLL(
            criterion=pfn_backend.criterion,
            logger=file_logger, device=device
        )

        context_sizes = cfg.context_sizes
        context_sizes = [math.floor(i * target_task_context_x.shape[0]) for i in context_sizes]

        kwargs = {}

        if 'inference_kwargs' in cfg.model.keys():
            kwargs.update(cfg.model.inference_kwargs)

        if not bool(prefix):
            prefix = (torch.empty(0, device=device), torch.empty(0, device=device))

        n_related = related_task_data.x.shape[1]
        evaluator.test_on_new_task(
            model=model,
            prefix_x=prefix[0],
            prefix_y=prefix[1],
            context_task_x=target_task_context_x.repeat(1, n_related, 1),
            context_task_y=target_task_context_y.repeat(1, n_related),
            query_task_x=target_task_query_x.repeat(1, n_related, 1),
            query_task_y=target_task_query_y.repeat(1, n_related),
            task_name=cfg.model.meta.name,
            step=0,
            context_sizes=context_sizes,
            # fwd kwargs
            **kwargs
        )

        # Quick Baselines --------------------
        # sanity check: what if we put in the current task as context.
        evaluator.test_on_new_task(
            model=pfn_backend,
            task_name=f'Baseline: complete target_task_x as context',

            context_task_x=target_task_context_x,
            context_task_y=target_task_context_y,
            query_task_x=target_task_query_x,
            query_task_y=target_task_query_y,


        )

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



