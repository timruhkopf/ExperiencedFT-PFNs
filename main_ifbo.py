import os
import time
import logging

from neps.search_spaces.parameter import Parameter
from typing import Any

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
from neps_run import neps_run

from dataset.batch_padded_pfn import parse_batch_for_padded_train_data
from src.utils.seeding import SeededRandomContext
from utils.dotdict import DotDict
from utils.filelogger_json import BufferedDictLogger

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

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_path="configs", config_name="base_ifbo", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f'Sweep dir: {Path.cwd()}')
    logger.info(OmegaConf.to_yaml(cfg))

    if hasattr(cfg.benchmark, "api"):
        delattr(cfg.benchmark.api, "step_size")

        benchmark: Benchmark = hydra.utils.instantiate(cfg.benchmark.api)  # type: ignore

    else:
        benchmark: Benchmark = hydra.utils.instantiate(cfg.benchmark.cls)

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
            isinstance(benchmark, TaskSetTabularBenchmark)
            and cfg.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        benchmark.meta = DotDict({})  # make it mutable
        benchmark.meta.name=''
        print(f"PREPROCESSING {benchmark.meta.name} with "
              f"'apply_user_prior_step_0_median_normalized'")
        drop_0_epoch = cfg.benchmark.get("drop_epoch_0", True)
        print(f"PREPROCESSING {benchmark.meta.name} with '{drop_0_epoch}'")
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=drop_0_epoch,
        )

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

    file_logger = BufferedDictLogger(
        file_path=Path.cwd() / 'results.jsonl', buffer_size=10,
    )

    if cfg.n_splits >= 2 and len(benchmark) >= 2:
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.split_seed)
        folds = kf.split(benchmark)
        folds = list(folds)
    else:
        raise ValueError(
            f'Not enough data to perform {cfg.n_splits}-folds. '
        )
    if hasattr(cfg, 'fold_idx'):
        # if fold_idx is specified, we only run that fold

        folds = [folds[cfg.fold_idx]]

    for fold, (train_ids, test_ids) in enumerate(folds):
        train_ids = [int(i) for i in train_ids]
        test_ids = [int(i) for i in test_ids]  # convert to int if needed

        # some debug option for fast execution
        if hasattr(cfg, 'target_idx'):
            # if target_idx is specified, we only run that target task
            test_ids = [test_ids[cfg.target_idx]]

        if hasattr(cfg, 'train_idx'):
            # if train_ids is specified, we only run that set of train tasks
            train_ids = [train_ids[i] for i in cfg.train_idx]

        for target_task in test_ids:

            if 'allocation_seeds' not in cfg.keys():
                allocation_seeds = [cfg.split_seed]
            else:
                allocation_seeds = cfg.allocation_seeds

            for allocation_seed in allocation_seeds:

                logger.info(f"Running task: target_task={target_task}, train_ids={train_ids},"
                            f"fold {fold}, allocation_seed={allocation_seed}")

                file_logger.postfix = {
                    'target_task': target_task,
                    'fold': fold,
                    'allocation_seed': allocation_seed,
                    'seed': cfg.seed
                }

                # "instantiate" the task and related task datasets (with no budget allocation yet)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)

                    if hasattr(benchmark, 'collect_task_split'):
                        # FIXME if for compat. reasons in debugging
                        benchmark.collect_task_split(target_id=target_task, train_ids=train_ids,
                                                     seed=allocation_seed, bnn_seed=cfg.seed)

                with (SeededRandomContext(allocation_seed)):
                    config = dict(
                        single_eval_pos=[1000] * (len(train_ids)),
                        alphas=[10 ** np.random.uniform(-4, -1) for _ in range(len(train_ids))],
                        **cfg.benchmark.sample_config if hasattr(cfg.benchmark, 'sample_config') else {},
                    )
                    # sample the dirichlet distributed data
                    if hasattr(benchmark, 'sample_batch'):
                        batch = benchmark.sample_batch(**config)

                        # parse the batch ---------------------------------------------
                        related_task_data = parse_batch_for_padded_train_data(batch)


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

                if 'nepsnevals' in cfg.keys():
                    max_evaluations_total = cfg.nepsnevals
                elif "mf" in cfg.algorithm and cfg.algorithm.mf:
                    max_evaluations_total = NEPS_MF_MAX_EVALS if cfg.algorithm.sh_based else (
                        NEPS_MF_EI_MAX_EVALS)
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
                        _table = preprocess_tabular(cfg.benchmark.meta.name, benchmark.table)
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
                        try:
                            table = table[list(set(table.columns).intersection(set(space.keys())))]
                            obj.pipeline_space.set_custom_grid_space(table[list(
                                space.hyperparameters.keys())],
                                                                     space)
                            if SET_BOUNDS_FROM_TABLE_FLAG:
                                obj = set_bounds_from_table(obj, table, space)
                        except KeyError as e:

                                logger.warning(
                                    f"Skipping config due to missing key 'linear_decay'. Table snapshot:\n{table}"
                                f"snapshot:\n{space.hyperparameters}")
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
                neps_dir = Path.cwd() / (f"neps_root_directory_{target_task}_{fold}"
                                         f"_{cfg.split_seed}_{cfg.seed}_{allocation_seed}")
                searcher_path = Path(__file__).parent / 'ifBO_icml2024/src/pfns_hpo/pfns_hpo/configs/algorithm/'
                if cfg.algorithm.name in [ 'dpl-neps-max',  'dyhpo-neps-v2']:
                    # TO FIX their damn paths!
                    searcher = cfg.algorithm.name
                    import tempfile
                    import yaml

                    with open(searcher_path /(searcher + '.yaml'), 'r') as file:
                        data = yaml.safe_load(file)

                        data['searcher_kwargs']['surrogate_model_args']['root_directory'] = \
                        neps_dir.name
                    tmpdir = tempfile.TemporaryDirectory(prefix="myrun_", suffix="_tmp")
                    searcher_path = Path(tmpdir.name)
                    searcher_path.mkdir(parents=True, exist_ok=True)
                    with open(searcher_path / searcher_path / (searcher + '.yaml'), 'w') as file:
                        yaml.safe_dump(data, file)

                elif cfg.algorithm.name in ['asha', 'hyperband', 'bohb']:
                    searcher= cfg.algorithm.name



                elif cfg.algorithm.searcher.surrogate_model in 'pfn' :
                    searcher = cfg.algorithm.name

                if   'surrogate_model' in cfg.algorithm.keys() and \
                        '_target_' in cfg.algorithm.surrogate_model.cls.keys():

                    searcher = hydra.utils.instantiate(
                        cfg.algorithm.searcher,
                        pipeline_space=pipeline_space,
                        # surrogate_model=surrogate_model
                    )


                    surrogate_model = hydra.utils.instantiate(
                        cfg.algorithm.surrogate_model.cls,
                        logger=file_logger,
                        device=device,
                        related_task_data=related_task_data
                    )


                    searcher.model_policy.surrogate_model.nn = surrogate_model
                    searcher.model_policy.surrogate_model_name = surrogate_model.__name__

                # -----------------------------------------------------------------------

                neps_run(
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

                    searcher_path=searcher_path,
                    overwrite_working_directory=OVERWRITE,
                    pre_load_hooks=[set_grid_table_space],  # crucial in allowing tabular grid access
                    post_run_summary=True,  # important for efficient plotting
                )
                file_logger.flush()

                logger.info(f"Finished run for fold={fold}, target_task={target_task}, "
                            f"train_ids={train_ids}, seed={cfg.seed}_{allocation_seed}")

                # if tmpdir exists, remove it
                if 'tmpdir' in locals():
                    tmpdir.cleanup()

                if "mf" in cfg.algorithm and cfg.algorithm.mf:
                    plotter = Plotter3D(
                        algorithm=cfg.algorithm.name,
                        benchmark=cfg.benchmark.name if 'name' in cfg.benchmark.keys() else
                        cfg.benchmark.meta.name,
                        experiment_group=cfg.experiment_group,
                        seed=cfg.seed
                    )
                    _df = pd.read_csv(
                        neps_dir / "summary_csv" / "config_data.csv",
                        float_precision="round_trip"
                    )
                    plotter.plot3D(data=_df, run_path=Path().cwd())

    logger.info(f"All runs finished, results saved to {Path.cwd()}")


if __name__ == '__main__':
    from src.utils.resolvers import *

    main()
