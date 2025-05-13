import os
import time
import logging
import warnings
from itertools import product

from neps.search_spaces.parameter import Parameter
from tqdm import tqdm
from typing import Any, List

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import hydra
from omegaconf import DictConfig

import neps
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace
import ConfigSpace as CS
from mfpbench import Benchmark
from mfpbench.taskset_tabular.benchmark import TaskSetTabularBenchmark

import ifbo
from ifbo.transformer import TransformerModel
from ifBO_icml2024.src.pfns_hpo.pfns_hpo.plot3D import Plotter3D
from ifBO_icml2024.src.pfns_hpo.pfns_hpo.run import process_mfpbench_trajectories, \
    process_taskset_mfpbench_with_step_0_prior, preprocess_tabular, set_bounds_from_table

from src.evaluation.meta_train_test_split import k_folds, folds_of_size
from src.model.batch_padded_pfn import parse_batch_for_padded_train_data
from src.utils.filelogger import BufferedFileLogger
from src.utils.seeding import SeededRandomContext

logger = logging.getLogger("main_ifbo")

# NOTE: If editing this, please look for MIN_SLEEP_TIME
# in `read_results.py` and change it there too
MIN_SLEEP_TIME = 10  # 10s hopefully is enough to simulate wait times for metahyper

# Use this environment variable to force overwrite when running
OVERWRITE = False  # bool(os.environ.get("MF_EXP_OVERWRITE", False))

print(f"{'=' * 50}\noverwrite={OVERWRITE}\n{'=' * 50}")

BOHB_MAX_EVALS = 10  # number of HB brackets
NEPS_MF_EI_MAX_EVALS = 1000  # number of total function evaluations for multi-fidelity
NEPS_MF_MAX_EVALS = 400  # number of total function evaluations for multi-fidelity
NEPS_SF_MAX_EVALS = 200  # number of total function evaluations for single-fidelity

SET_BOUNDS_FROM_TABLE_FLAG = False  # if True, sets search space bounds from table values


@hydra.main(config_path="configs", config_name="base_ifbo", version_base="1.1")
def main(cfg: DictConfig):
    """
    This main function is basically the meta-task version of
    ifbo_icml2024/src/pfns_hpo/pfns_hpo/run.py run_neps
    :param cfg:
    :return:
    """

    import shutil
    logger.info(f'Sweep dir: {Path.cwd()}')
    logger.info(OmegaConf.to_yaml(cfg))

    # TODO: for fair experimentation, make this a benchmark property overriding algo
    if hasattr(cfg.benchmark, "api"):
        delattr(cfg.benchmark.api, "step_size")

        # TODO: @TIM make MFPBENCHPRIOR a proper benchmark class and add api to config
        #  lcbench-126026.yaml
        #  pd1-tabular-cifar10_wideresnet_256.yaml
        #  taskset-tabular-nlp-1-4p.yaml
        #  as example for the benchmark.api

        benchmark: Benchmark = hydra.utils.instantiate(cfg.benchmark.api)  # type: ignore

    else:
        benchmark: Benchmark = hydra.utils.instantiate(cfg.benchmark.cls)

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
            isinstance(benchmark, TaskSetTabularBenchmark)
            and cfg.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        print(f"PREPROCESSING {benchmark.meta.name} with "
              f"'apply_user_prior_step_0_median_normalized'")
        drop_0_epoch = cfg.benchmark.get("drop_epoch_0", True)
        print(f"PREPROCESSING {benchmark.meta.name} with '{drop_0_epoch=}'")
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=drop_0_epoch,
        )

    # OUR CODE to sample the related tasks of the benchmark and inform the model about them ahead
    # of the actual deployment in the MFHPO scenario ---------------------------
    # setting up the reference model:
    logger.info(f'Sweep dir: {Path.cwd()}')
    logger.info(f"Running with config: \n {OmegaConf.to_yaml(cfg, resolve=True)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device is None \
        else torch.device(cfg.device)
    ftpfn = ifbo.surrogate.FTPFN(
        target_path=Path(cfg.repo_root) / 'data' / '.model',
        version="0.0.1",
        device=device
    )
    pfn_backend: TransformerModel = ftpfn.model
    criterion = pfn_backend.criterion.to(device)

    file_logger = BufferedFileLogger(
        file_name='results.csv',
        file_path='.',
        buffer_size=100,
        header=["metric", "value", 'global_step', "context_size", "task"],
        postfix=["target_task", "train_ids", "seed"]
    )

    # Generate and select the folds (meta-train-test splits)
    all_train_ids, test_ids = train_test_split(
        # FIXME: default is just for compatability reasons in debug
        list(range(len(benchmark))) if hasattr(benchmark, '__len__') else list(range(100)),
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
    # TODO for loop over the alpha repetitions of the context tasks
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

            if hasattr(benchmark, 'collect_task_split'):
                # FIXME if for compat. reasons in debugging
                benchmark.collect_task_split(target_id=target_task, train_ids=train_ids)

        with (SeededRandomContext(seed)):
            config = dict(
                single_eval_pos=[500] * (len(train_ids) + 1),
                alphas=[10 ** np.random.uniform(-4, -1) for _ in range(len(train_ids) + 1)],
                **cfg.benchmark.sample_config if hasattr(cfg.benchmark, 'sample_config') else {}
            )
            # sample the dirichlet distributed data
            if hasattr(benchmark, 'sample_batch'):
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

                for k in ['x', 'y', 'query_x', 'query_y']:
                    if hasattr(related_task_data, k) and isinstance(
                            related_task_data.__getattribute__(k), torch.Tensor):
                        related_task_data.__setattr__(k,related_task_data.__getattribute__(k).to(device))



        # --------------------------------------------------------------------------

        # CRUCIAL check to determine if the benchmark is tabular (list of configs)
        bench_is_tabular = (
            True if hasattr(cfg.benchmark, "tabular") and cfg.benchmark.tabular else False
        )

        # TODO collect_task_split, sample_batch, parse and pass to the model for "train"

        def run_pipeline(previous_pipeline_directory: Path, **config: Any) -> dict:
            start = time.time()
            if benchmark.fidelity_name in config:
                fidelity = config.pop(benchmark.fidelity_name)
            else:
                fidelity = benchmark.fidelity_range[1]

            if bench_is_tabular:  # declared in parent scope
                # IMPORTANT to handle tabular benchmarks to query using only IDs
                # if "tabular" in cfg.benchmark.name:
                config = int(config["id"])
                # TODO: handle other tabular benchmarks

            full_trajectory = benchmark.trajectory(config)
            trajectory_to_query = [r for r in full_trajectory if r.fidelity <= fidelity]

            result = trajectory_to_query[-1]
            max_fidelity_result = full_trajectory[-1]

            # best seen till the fidelity specified
            _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(
                trajectory_to_query)
            # best seen ever for the config till max fidelity
            _, min_valid_ever, min_test_ever = process_mfpbench_trajectories(full_trajectory)

            end = time.time()

            return {
                "loss": result.error,
                "cost": result.cost,
                "info_dict": {
                    "cost": result.cost,
                    "val_score": result.val_score,
                    "test_score": result.test_score,
                    "fidelity": result.fidelity,
                    "continuation_fidelity": None,
                    "start_time": start,
                    "end_time": end,  # + fidelity,
                    "max_fidelity_loss": float(max_fidelity_result.error),
                    "max_fidelity_cost": float(max_fidelity_result.cost),
                    "process_id": os.getpid(),
                    "min_valid_seen": min_valid_seen,
                    "min_test_seen": min_test_seen,
                    "min_valid_ever": min_valid_ever,
                    "min_test_ever": min_test_ever,
                    "learning_curve": _result["valid"],
                    "learning_curves": _result,
                },
            }

        pipeline_space = {
            "search_space": benchmark.space
        }
        lower, upper, _ = benchmark.fidelity_range
        fidelity_name = benchmark.fidelity_name
        if "mf" in cfg.algorithm and cfg.algorithm.mf:
            if isinstance(lower, float):
                fidelity_param = neps.FloatParameter(
                    lower=lower, upper=upper, is_fidelity=True
                )
            else:
                fidelity_param = neps.IntegerParameter(
                    lower=lower, upper=upper, is_fidelity=True
                )
            pipeline_space = {**pipeline_space, **{fidelity_name: fidelity_param}}
            logger.info(f"Using fidelity space: \n {fidelity_param}")
        logger.info(f"Using search space: \n {pipeline_space}")

        if "mf" in cfg.algorithm and cfg.algorithm.mf:
            max_evaluations_total = NEPS_MF_MAX_EVALS if cfg.algorithm.sh_based else NEPS_MF_EI_MAX_EVALS
        else:
            max_evaluations_total = NEPS_SF_MAX_EVALS

        # placeholder pre_load hook
        def set_grid_table_space(
                obj,
                **kwargs
        ) -> Any:
            return obj

        # snippet to handle tabular benchmarks
        if bench_is_tabular:
            # extracting and processing the tabular data and raw space
            #
            if hasattr(benchmark, 'sample_batch'):
                _table = preprocess_tabular(benchmark.target_benchmark.name, benchmark.table)
            else:
                _table = preprocess_tabular(cfg.benchmark.name, benchmark.table)
            # updates the pipeline_space to be only config IDs mapping to tabular data
            pipeline_space = {
                "id": neps.IntegerParameter(
                    lower=_table.index.min(), upper=_table.index.max()
                )
            }
            # include the fidelity in the spaces
            if "mf" in cfg.algorithm and cfg.algorithm.mf:
                pipeline_space.update({fidelity_name: fidelity_param})
                # include fidelity in the raw benchmark space
                if isinstance(benchmark.space, CS.ConfigurationSpace):
                    _space = pipeline_space_from_configspace(benchmark.space)
                    _space.update({fidelity_name: fidelity_param})
                    benchmark.space = SearchSpace(**_space)
                elif isinstance(benchmark.space, dict):
                    benchmark.space.update({fidelity_name: fidelity_param})
                elif isinstance(benchmark.space, SearchSpace):
                    benchmark.space.add_hyperparameter(name=fidelity_name, hp=fidelity_param)
                else:
                    raise ValueError("Unknown benchmark space type!")
            else:
                benchmark.space = SearchSpace(**pipeline_space_from_configspace(benchmark.space))

            # (re-)defines a pre_load_hook to handle tabular data explicitly
            # CRUCIAL for tabular benchmarks with a fixed list of configs
            def set_grid_table_space(
                    # overwrites the placeholder in the parent scope
                    obj,
                    table: pd.DataFrame | pd.Series = _table,
                    space: CS.ConfigurationSpace = benchmark.space,
            ) -> Any:
                # both table and space are required to handle tabular spaces
                hps = list(set(space.keys()).intersection(set(table.columns)))

                obj.pipeline_space.set_custom_grid_space(table[hps], space)
                if SET_BOUNDS_FROM_TABLE_FLAG:
                    obj = set_bounds_from_table(obj, table, space)
                return obj
        # end of tabular check block

        print("MAX EVALUATIONS:", max_evaluations_total)

        # ---------------------------------------------------
        # Manual edit of the search space accoding to  neps.api._run_args l 324
        if hasattr(benchmark, 'sample_batch'):
            try:
                # Support pipeline space as ConfigurationSpace definition
                if isinstance(pipeline_space, CS.ConfigurationSpace):
                    pipeline_space = pipeline_space_from_configspace(pipeline_space)

                # Support pipeline space as mix of ConfigurationSpace and neps parameters
                new_pipeline_space: dict[str, Parameter] = dict()
                for key, value in pipeline_space.items():
                    if isinstance(value, CS.ConfigurationSpace):
                        config_space_parameters = pipeline_space_from_configspace(value)
                        new_pipeline_space = {**new_pipeline_space, **config_space_parameters}
                    else:
                        new_pipeline_space[key] = value
                pipeline_space = new_pipeline_space

                # Transform to neps internal representation of the pipeline space
                pipeline_space = SearchSpace(**pipeline_space)
            except TypeError as e:
                message = f"The pipeline_space has invalid type: {type(pipeline_space)}"
                raise TypeError(message) from e

        # ---------------------------------------------------

        if cfg.algorithm.searcher.surrogate_model not in ['pfn', 'dpl', 'deep_gp']:
            searcher = cfg.algorithm.name
        else:

            searcher = hydra.utils.instantiate(
                cfg.algorithm.searcher,
                pipeline_space=pipeline_space,
                # surrogate_model=surrogate_model
            )

            if 'surrogate_model' in cfg.algorithm.keys():
                surrogate_model = hydra.utils.instantiate(
                    cfg.algorithm.surrogate_model.cls,
                    logger=file_logger,
                    device=device,
                    related_task_data=related_task_data
                )

                cfgmodel = cfg.algorithm.surrogate_model

                train_config = dict(
                    query_task_x=target_task_query_x,
                    query_task_y=target_task_query_y
                )
                if 'train_call' in cfgmodel.keys():
                    train_config.update(cfgmodel.train_call)

                # distillation will return a prefix, pfn_mixture won't
                surrogate_model.train(**train_config)

                searcher.model_policy.surrogate_model.nn = surrogate_model
                searcher.model_policy.surrogate_model_name = surrogate_model.__name__

        # -----------------------------------------------------------------------
        neps_dir = f"neps_root_directory_{target_task}_{train_ids}_{seed}"
        neps.run(
            run_pipeline=run_pipeline,
            pipeline_space=pipeline_space,
            root_directory=neps_dir,
            # TODO: figure out how to pass runtime budget and if metahyper internally
            #  calculates continuation costs to subtract from optimization budget
            # **budget_args,
            max_evaluations_total=max_evaluations_total,

            # FIXME: backward compat: hasattr(cfg.algorithm, 'searcher') for
            searcher=searcher,

            # FIXME: add in a searcher instantiation (BaseOptimizer subclass), that will also get
            #  the benchmark instance as info

            searcher_path=Path(__file__).parent / 'ifBO_icml2024' / 'src' / 'pfns_hpo' /
                          'pfns_hpo' / 'configs' / "algorithm",
            overwrite_working_directory=OVERWRITE,
            pre_load_hooks=[set_grid_table_space],  # crucial in allowing tabular grid access
            post_run_summary=True,  # important for efficient plotting
        )

        if "mf" in cfg.algorithm and cfg.algorithm.mf:
            plotter = Plotter3D(
                algorithm=cfg.algorithm.name,
                benchmark=cfg.benchmark.name if 'name' in cfg.benchmark.keys() else
                cfg.benchmark.meta.name,
                experiment_group=cfg.experiment_group,
                seed=cfg.seed
            )
            _df = pd.read_csv(
                Path().cwd() / neps_dir / "summary_csv" / "config_data.csv",
                float_precision="round_trip"
            )
            plotter.plot3D(data=_df, run_path=Path().cwd())

    file_logger.close()


if __name__ == '__main__':
    from src.utils.resolvers import *

    main()
