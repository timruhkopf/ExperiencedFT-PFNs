import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ifBO_icml2024.src.pfns_hpo.pfns_hpo.regret_plot import calculate_continuations
from ifBO_icml2024.src.pfns_hpo.pfns_hpo.utils.plotting_utils import calc_bounds_per_benchmark, \
    normalize_and_calculate_regrets, reorder_for_aggregated_benchmark_plots, group_run_dataframes, \
    get_plot_styles
from ifBO_icml2024.src.pfns_hpo.pfns_hpo.utils.plotting_utils import PLOT_SIZE
from src.utils.read_neps import parse_and_save


def main(
        basedir,
        benchmarks=None,
        algorithms=None,
        seeds=None,
        continuations=True,
        wallclock=False,
        overhead=False,
        log_x=False, log_y=True, x_range=(1, 1000),
        normalize="baseline",
        filename=None,
        plot_aggregate=True,
        plot_per_benchmark=False,
        color_map={},
        marker_map={},
        label_map={},
        output_path=None
):
    if output_path is None:
        output_path = Path(basedir) / "plots"
    df = parse_and_save(
        root_dir=basedir,
        # TODO move this
        keys=["experiment_name", "algorithm.surrogate_model.meta.name", "algoname",
              "benchmark.meta.name", "split_seed", "benchmark.cls.seed", ],
        file_pattern="config_data.csv",
    )

    plot_data = {
        f'{bench}_{fold}_{task}_{allocation_seed}': {
            algo: {
                int(seed): df[
                    (df["benchmark.meta.name"] == bench) &
                    (df["target_task"] == task) &
                    (df["fold"] == fold) &
                    (df["algoname"] == algo) &
                    (df["seed"] == seed) &
                    (df['split_seed'] == split_seed) &
                    (df["allocation_seed"] == allocation_seed)
                    ]  # .sort_values("step").reset_index(drop=True)
                for seed in (
                    seeds if seeds is not None else
                    df[(df["benchmark.meta.name"] == bench) & (df["algoname"] == algo)][
                        "seed"].unique()
                )

            }
            for algo in (algorithms if algorithms is not None else df["algoname"].unique())
        }
        for bench in (benchmarks if benchmarks is not None else df["benchmark.meta.name"].unique())
        for fold in df['fold'].unique()
        for task in df['target_task'].unique()
        for allocation_seed in df['allocation_seed'].unique()
        for split_seed in df['split_seed'].unique()
    }
    # Removing empty entries
    plot_data = {
        key1: {
            key2: {
                seed: df for seed, df in seed_dict.items() if not df.empty
            }
            for key2, seed_dict in algo_dict.items() if
            any(not df.empty for df in seed_dict.values())
        }
        for key1, algo_dict in plot_data.items() if any(
            any(not df.empty for df in seed_dict.values()) for seed_dict in algo_dict.values()
        )
    }

    # Normalizing incumbents
    name_normalization = ""
    if normalize in ["baseline", "benchmark", "optimum"]:
        if normalize == "baseline":
            bounds = calc_bounds_per_benchmark(plot_data)
            name_normalization = "normBaseline"
        elif normalize == "benchmark":
            benchmark_name = benchmarks[0].split("-")[0]
            with open(basedir / ".." / f"benchmarks_bounds/{benchmark_name}.json", "r") as f:
                bounds = json.load(f)
            name_normalization = "normBenchmark"
        plot_data = normalize_and_calculate_regrets(plot_data, bounds)
    elif normalize == "none":
        name_normalization = "noNorm"
    else:
        raise ValueError(f"Invalid normalization: {normalize}")

    # Plotting aggregated plots
    if plot_aggregate:
        print("Plotting aggregated plot...")
        # reorder data
        get_aggregated_plot_style(
            plot_data.copy(),
            output_path,
            filename="aggregated" if filename is None else f"aggregated_{filename}_{name_normalization}",
            log_x=log_x,
            log_y=log_y,
            x_range=x_range,
            color_map=color_map,
            marker_map=marker_map,
            # marker_args=DEFAULT_MARKER_KWARGS,
            label_map=label_map,
            wallclock=wallclock,
            overhead=overhead,
        )


def get_aggregated_plot_style(
        plot_data: dict,
        output_path: Path,
        filename: str,
        column_of_interest: str = "inc_loss",
        log_x: bool = False,
        log_y: str = False,
        x_range: tuple = None,
        color_map: dict = {},
        marker_map: dict = {},
        marker_args: dict = {},
        label_map: dict = {},
        wallclock: bool = False,
        overhead: bool = False,
        analysis: bool = False,

) -> None:
    """Plots a single plot aggregating performance of each algorithm across benchmarks."""
    plot_data = reorder_for_aggregated_benchmark_plots(plot_data)
    # processing data for plotting
    algo_perf = dict()  # nothing to do with https://arxiv.org/abs/2306.07179
    for algo, algo_data in plot_data.items():
        algo_perf[algo] = dict()
        seeds = []
        for seed, seed_data in algo_data.items():
            seeds.append(seed)
            # collecting the mean score across benchmarks
            algo_perf[algo][seed], _ = group_run_dataframes(
                list(seed_data.values()),
                column_of_interest=column_of_interest,
                analysis=analysis,
                x_range=x_range,
            )
        # averaging score across seeds
        algo_perf[algo]["mean"], algo_perf[algo]["sem"] = (
            group_run_dataframes(
                list(algo_perf[algo].values()),
                column_of_interest=column_of_interest,
                analysis=analysis,
                x_range=x_range,
            )
        )
        # removing seed keys
        _ = [algo_perf[algo].pop(_seed) for _seed in seeds]
    # Define a list of line styles and markers
    if color_map is None or marker_map is None:
        l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)
    else:
        l_colors = color_map
        l_markers = marker_map
    # l_colors, l_line_styles, l_markers = get_plot_styles(algo_perf)

    # Do the plotting
    plt.clf()

    algo_order = [(algo, algo_data["mean"].values[-1]) for algo, algo_data in algo_perf.items()]
    algo_order.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(1, 1, figsize=(PLOT_SIZE * 2, int(PLOT_SIZE * 1.5)))

    for i, (algo, _) in enumerate(algo_order):
        algo_data = algo_perf[algo]
        ax.plot(
            algo_data["mean"].index.values,
            algo_data["mean"].values,
            color=l_colors[algo] if algo in l_colors else f"C{i}",
            # linestyle=l_line_styles[algo],
            marker=l_markers[algo] if algo in l_markers else "o",
            # markersize=6,
            markevery=max(1, int(len(algo_data["mean"].index.values) / 15)),
            # fillstyle='none',
            **marker_args,
            label=algo if algo not in label_map else label_map[algo],
        )
        ax.fill_between(
            algo_data["mean"].index.values,
            algo_data["mean"].values - algo_data["sem"].values,
            algo_data["mean"].values + algo_data["sem"].values,
            facecolor=l_colors[algo] if algo in l_colors else f"C{i}",
            alpha=0.1,
            step="post"
        )
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if x_range is not None:
        ax.set_xlim(*x_range)

    if wallclock:
        fig.supxlabel("Wallclock time (in s)")
    elif not wallclock and overhead:
        fig.supxlabel("Only overhead time (in s)")
    else:
        fig.supxlabel("Total epochs spent")
    ax.set_ylabel("Normalized regret")  #\
        # if not analysis else ax.set_ylabel(
        # ANALYSIS_Y_LABEL[column_of_interest])

    # Move the legend to the right side of the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # fig.suptitle("Aggregate results across benchmarks")
    plt.tight_layout()

    target = output_path / f"{filename}.png"
    plt.savefig(target, bbox_inches='tight')
    print(f"\nPlot saved as {target}\n")


if __name__ == '__main__':
    import fire

    fire.Fire(main)
