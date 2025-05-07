import logging
import os
import time
from pathlib import Path
from typing import Any, List
import ConfigSpace as CS
import hydra
from omegaconf import DictConfig
import pandas as pd

from mfpbench import Benchmark
from mfpbench.taskset_tabular.benchmark import TaskSetTabularBenchmark

import neps
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace

from src.ifBO_icml2024.src.pfns_hpo.pfns_hpo.plot3D import Plotter3D
from src.ifBO_icml2024.src.pfns_hpo.pfns_hpo.run import process_mfpbench_trajectories, \
    process_taskset_mfpbench_with_step_0_prior, preprocess_tabular

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

    # TODO: for fair experimentation, make this a benchmark property overriding algo
    delattr(cfg.benchmark.api, "step_size")

    # TODO: @TIM make MFPBENCHPRIOR a proper benchmark class and add api to config
    #  lcbench-126026.yaml
    #  pd1-tabular-cifar10_wideresnet_256.yaml
    #  taskset-tabular-nlp-1-4p.yaml
    #  as example for the benchmark.api

    benchmark: Benchmark = hydra.utils.instantiate(cfg.benchmark.api)  # type: ignore

    # Maybe apply step 0 median normalization, read docstring of func for more
    if (
            isinstance(benchmark, TaskSetTabularBenchmark)
            and cfg.benchmark.get("apply_user_prior_step_0_median_normalized", False) is True
    ):
        print(f"PREPROCESSING {benchmark.name} with 'apply_user_prior_step_0_median_normalized'")
        drop_0_epoch = cfg.benchmark.get("drop_epoch_0", True)
        print(f"PREPROCESSING {benchmark.name} with '{drop_0_epoch=}'")
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark,
            drop_step_0=drop_0_epoch,
        )

    # CRUCIAL check to determine if the benchmark is tabular (list of configs)
    bench_is_tabular = (
        True if hasattr(cfg.benchmark, "tabular") and cfg.benchmark.tabular else False
    )

    def run_pipeline(previous_pipeline_directory: Path, **config: Any) -> dict:
        start = time.time()
        if benchmark.fidelity_name in config:
            fidelity = config.pop(benchmark.fidelity_name)
        else:
            fidelity = benchmark.fidelity_range[1]

        if bench_is_tabular:  # declared in parent scope
            # IMPORTANT to handle tabular benchmarks to query using only IDs
            if "tabular" in cfg.benchmark.name:
                config = int(config["id"])
            # TODO: handle other tabular benchmarks

        full_trajectory = benchmark.trajectory(config)
        trajectory_to_query = [r for r in full_trajectory if r.fidelity <= fidelity]

        result = trajectory_to_query[-1]
        max_fidelity_result = full_trajectory[-1]

        # best seen till the fidelity specified
        _result, min_valid_seen, min_test_seen = process_mfpbench_trajectories(trajectory_to_query)
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
            obj.pipeline_space.set_custom_grid_space(table, space)
            if SET_BOUNDS_FROM_TABLE_FLAG:
                obj = set_bounds_from_table(obj, table, space)
            return obj
    # end of tabular check block

    print("MAX EVALUATIONS:", max_evaluations_total)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="neps_root_directory",
        # TODO: figure out how to pass runtime budget and if metahyper internally
        #  calculates continuation costs to subtract from optimization budget
        # **budget_args,
        max_evaluations_total=max_evaluations_total,
        searcher=cfg.algorithm.name,

        # FIXME: @TIM add in a searcher instantiation (BaseOptimizer subclass), that will also get
        #  the benchmark instance as info

        # hydra.utils.instantiate(args.algorithm.searcher, _partial_=True),
        searcher_path=Path(__file__).parent / "configs" / "algorithm",
        overwrite_working_directory=OVERWRITE,
        pre_load_hooks=[set_grid_table_space],  # crucial in allowing tabular grid access
        post_run_summary=True,  # important for efficient plotting
    )


    if "mf" in cfg.algorithm and cfg.algorithm.mf:
        plotter = Plotter3D(
            algorithm=cfg.algorithm.name,
            benchmark=cfg.benchmark.name,
            experiment_group=cfg.experiment_group,
            seed=cfg.seed
        )
        _df = pd.read_csv(
            Path().cwd() / "neps_root_directory" / "summary_csv" / "config_data.csv",
            float_precision="round_trip"
        )
        plotter.plot3D(data=_df, run_path=Path().cwd())

if __name__ == '__main__':
    from src.utils.resolvers import *

    main()