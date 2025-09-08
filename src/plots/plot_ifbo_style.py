import json
from pathlib import Path
from typing import Dict, List

from scipy.stats import sem

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = False

import seaborn as sns

from ifBO_icml2024.src.pfns_hpo.pfns_hpo.regret_plot import calculate_continuations
from ifBO_icml2024.src.pfns_hpo.pfns_hpo.utils.plotting_utils import calc_bounds_per_benchmark, \
    normalize_and_calculate_regrets, reorder_for_aggregated_benchmark_plots, group_run_dataframes, \
    get_plot_styles, smooth_gaussian
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
        log_x=False,
        log_y=True, x_range=(1, 1000),
        normalize="baseline",
        filename=None,
        plot_aggregate=True,
        plot_per_benchmark=False,
        color_map={},
        marker_map={},
        label_map={},
        median=False,
        output_path=None
):
    if output_path is None:
        output_path = Path(basedir) / "plots"
        output_path.mkdir(parents=True, exist_ok=True)
    df = parse_and_save(
        root_dir=basedir,
        # TODO move this
        keys=["experiment_name", "algorithm.surrogate_model.meta.name", "algoname",
              "benchmark.meta.name", "split_seed", "benchmark.cls.seed", ],
        file_pattern="config_data.csv",
    )
    # fixme: uggly code ahead: make a product call and check if df.empty to avoid postprocessing
    #  and nested loops
    # plot_data = {
    #     f'{bench}_{fold}_{task}_{allocation_seed}': {
    #         algo: {
    #             int(seed): df[
    #                 (df["benchmark.meta.name"] == bench) &
    #                 (df["target_task"] == task) &
    #                 (df["fold"] == fold) &
    #                 (df["algoname"] == algo) &
    #                 (df["seed"] == seed) &
    #                 (df['split_seed'] == split_seed) &
    #                 (df["allocation_seed"] == allocation_seed)
    #                 ]  # .sort_values("step").reset_index(drop=True)
    #             for seed in (
    #                 seeds if seeds is not None else
    #                 df[(df["benchmark.meta.name"] == bench) & (df["algoname"] == algo)][
    #                     "seed"].unique()
    #             )
    #
    #         }
    #         for algo in (algorithms if algorithms is not None else df["algoname"].unique())
    #     }
    #     for bench in (benchmarks if benchmarks is not None else df["benchmark.meta.name"].unique())
    #     for fold in df['fold'].unique()
    #     for task in df['target_task'].unique()
    #     for allocation_seed in df['allocation_seed'].unique()
    #     for split_seed in df['split_seed'].unique()
    # }

    #
    # (f"neps_root_directory_{target_task}_{fold}_{cfg.split_seed}_{cfg.seed}_{allocation_seed}")
    # x_var = "cumsum_fidelity"
    # y_var = "inc_loss"
    #
    # facet_var = ['benchmark.meta.name']
    # hue_var = ['algoname']
    # norm_vars = ['benchmark.meta.name', 'target_task', 'split_seed']
    # aggregate_var = ['target_task', 'split_seed', 'fold', 'allocation_seed']
    #
    # def min_max_normalize_group(df, norm_vars, target_var):
    #     min_vals = df.groupby(norm_vars)[target_var].transform('min')
    #     max_vals = df.groupby(norm_vars)[target_var].transform('max')
    #     df[target_var + '_minmaxnorm'] = (df[target_var] - min_vals) / (max_vals - min_vals)
    #     return df
    #
    # # Apply min-max normalization
    #
    # data = min_max_normalize_group(df, norm_vars, y_var)
    #
    # agg_vars = [x_var, 'benchmark.meta.name', 'algoname']
    # agg_func = {
    #     y_var + '_minmaxnorm': [
    #         ('mean', 'mean'),
    #         ('sem', 'sem'),
    #         ('median', 'median'),
    #         ('lower', lambda x: np.percentile(x, 25)),
    #         ('upper', lambda x: np.percentile(x, 75))
    #     ]
    # }
    # agg_df = data.groupby(agg_vars).agg(agg_func)
    # agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    # agg_df = agg_df.reset_index()
    #
    #
    #
    #
    # sns.set(style="whitegrid")
    # fig, axes = plt.subplots(
    #     nrows=1, ncols=len(data['benchmark.meta.name'].unique()),
    #     figsize=(16, 6),
    #     sharey=True
    # )
    #
    # if len(data['benchmark.meta.name'].unique()) == 1:
    #     axes = [axes]
    #
    # if median:
    #     middle = y_var + '_minmaxnorm_median'
    #     lower = y_var + '_minmaxnorm_lower'
    #     upper = y_var + '_minmaxnorm_upper'
    # else:
    #     middle = y_var + '_minmaxnorm_mean'
    #     lower = y_var + '_minmaxnorm_sem'
    #     upper = y_var + '_minmaxnorm_sem'
    #
    #     # deduplicate entries
    # agg_df = agg_df.drop_duplicates(subset=facet_var + hue_var + [middle])
    #
    # palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999',
    #            '#e41a1c', '#dede00', '#66c2a5']
    # line_styles = ['-', '--', '-.', ':']
    #
    # algorithms = list(agg_df['algoname'].unique())
    # color_map = {alg: palette[i % len(palette)] for i, alg in enumerate(algorithms)}
    # markers = ['o', 's', '^', 'D', 'x', '*', 'P', 'H', 'v', '<', '>']
    # marker_map = {alg: markers[i % len(markers)] for i, alg in enumerate(algorithms)}
    #
    # for ax, (bench_name, group) in zip(axes, agg_df.groupby('benchmark.meta.name')):
    #     for algoname, sub_group in group.groupby('algoname'):
    #
    #         color = color_map[algoname]
    #         marker = marker_map[algoname]
    #         ax.plot(
    #             sub_group[x_var],
    #             sub_group[middle],
    #             label=algoname,
    #             color=color,
    #             marker=marker,
    #             markersize=5,
    #         )
    #         ax.fill_between(
    #             sub_group[x_var],
    #             sub_group[middle] - sub_group[lower] if not median else sub_group[lower],
    #             sub_group[middle] + sub_group[upper] if not median else sub_group[upper],
    #             color=color,
    #             alpha=0.3
    #         )
    #     ax.set_title(f'Benchmark: {bench_name}')
    #     ax.set_xlabel('Fidelity')
    #     ax.set_ylabel(f'Normalized Incumbent Loss')
    #     ax.legend(title='Algorithm')
    #     ax.set_xlim(0, 1000)
    #     if log_y:
    #         ax.set_yscale("log")
    #
    # plt.tight_layout()
    # plt.show()
    #
    # target = output_path / f"aggregated_minmax_normalized.png"
    # plt.savefig(target, bbox_inches='tight')
    # print(f"\nPlot saved as {target}\n")

    groups = df.groupby([
        "benchmark.meta.name", "fold", "target_task", "allocation_seed", "split_seed", "algoname",
        "seed"
    ], sort=False)

    plot_data = {}

    for group_keys, group_df in groups:
        bench, fold, task, allocation_seed, split_seed, algo, seed = group_keys
        key = f"{bench}_{fold}_{task}_{allocation_seed}"
        plot_data.setdefault(key, {}).setdefault(algo, {})[int(seed)] = group_df
    #
    # # Removing empty entries
    # plot_data = {
    #     key1: {
    #         key2: {
    #             seed: df for seed, df in seed_dict.items() if not df.empty
    #         }
    #         for key2, seed_dict in algo_dict.items() if
    #         any(not df.empty for df in seed_dict.values())
    #     }
    #     for key1, algo_dict in plot_data.items() if any(
    #         any(not df.empty for df in seed_dict.values()) for seed_dict in algo_dict.values()
    #     )
    # }

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


def group_run_dataframes_ppfn(
        df_list: List[pd.DataFrame],
        column_of_interest: str = "inc_loss",
        aggregate_by: str = "cumsum_fidelity",
        **kwargs
):
    """Given a list of dataframes, collates them based on the index."""
    assert len(df_list), "Empty list! Needs at least one element as a pd.DataFrame!"
    list_status = all(isinstance(element, (pd.DataFrame, pd.Series)) for element in df_list)
    assert list_status, "All elements in the list are not pandas data types!"

    # sets the index to be the cumulative fidelities and filters for the chosen columns
    if isinstance(df_list[0], pd.DataFrame):
        df_list = [
            _df.set_index(aggregate_by).loc[:, column_of_interest] for _df in df_list
        ]
        # if a list of pd.Series, likely a grouping was performed already
        # TODO: more strict check? check all the _df have the same length.
        # assert all(df_list[0].shape[0] == _df.shape[0] for _df in df_list[1:])

    # filter for the range of interest
    x_range = kwargs.get("x_range", None)
    if x_range is not None:
        df_list = [df.loc[x_range[0]:x_range[1]] for df in df_list]
    if sum([any(_df.index.values > x_range[1]) for _df in df_list]):
        raise ValueError("Some dataframes have indices greater than the specified range!")
    # get the last/max index
    max_index = max(set().union(*[set(df.index) for df in df_list]))
    # retains only the rows where incumbent values are updated, helps save memory
    df_list = [df[df.diff().ne(0)] for df in df_list]

    # remove spurious NaNs from index
    df_list = [df.loc[df.index.dropna()] for df in df_list]

    # removes duplicate indices
    df_list = [df.loc[~df.index.duplicated(keep="first")] for df in df_list]

    # collects the unique values seen for the x-axis
    union_index = pd.Index(set([max_index]).union(*[set(df.index) for df in df_list])).sort_values()

    # equalizes all data frames to have this unique list of x-axis values accounted for
    df_values = np.array([
        df.reindex(union_index, method='ffill').sort_index().values for df in df_list
    ])

    # smooth for analysis
    if kwargs.get("analysis", False):
        df_values = np.array(smooth_gaussian(df_values, sigma=0.35))

    # if kwargs.get("nanmean", False):
    #     df_values_ = np.nan_to_num(df_values, nan=1)
    #     mean_df = pd.Series(df_values_.mean(axis=0), index=union_index).sort_index()
    #     sem_df = pd.Series(sem(df_values_, axis=0), index=union_index).sort_index()
    # else:
    #     mean_df = pd.Series(df_values.mean(axis=0), index=union_index).sort_index()
    #     sem_df = pd.Series(sem(df_values, axis=0), index=union_index).sort_index()

    median_values = np.median(df_values, axis=0)
    median_df = pd.Series(median_values, index=union_index).sort_index()
    lower_quantile_values = np.quantile(df_values, 0.25, axis=0)
    upper_quantile_values = np.quantile(df_values, 0.75, axis=0)

    lower_quantile_df = pd.Series(lower_quantile_values, index=union_index).sort_index()
    upper_quantile_df = pd.Series(upper_quantile_values, index=union_index).sort_index()
    return median_df, (lower_quantile_df, upper_quantile_df)


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
        median: bool = False,

) -> None:
    """Plots a single plot aggregating performance of each algorithm across benchmarks."""
    if median:
        func = group_run_dataframes_ppfn
    else:
        func = group_run_dataframes

    plot_data = reorder_for_aggregated_benchmark_plots(plot_data)
    # processing data for plotting
    algo_perf = dict()  # nothing to do with https://arxiv.org/abs/2306.07179
    for algo, algo_data in plot_data.items():
        algo_perf[algo] = dict()
        seeds = []
        for seed, seed_data in algo_data.items():
            seeds.append(seed)
            # collecting the mean score across benchmarks
            algo_perf[algo][seed], _ = func(
                list(seed_data.values()),
                column_of_interest=column_of_interest,
                analysis=analysis,
                x_range=x_range,
            )

        # averaging score across seeds
        algo_perf[algo]["mean"], algo_perf[algo]["sem"] = (
            func(
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
    # Choose a colormap, e.g., 'tab20', 'nipy_spectral', etc.
    cmap = plt.get_cmap('tab20')
    num_algos = len(algo_order)

    # Generate a list of colors (RGBA tuples)
    colors = [cmap(i / num_algos) for i in range(num_algos)]
    for i, (algo, _) in enumerate(algo_order):
        algo_data = algo_perf[algo]
        ax.plot(
            algo_data["mean"].index.values,
            algo_data["mean"].values,
            color=colors[i],  # l_colors[algo] if algo in l_colors else f"C{i}",
            # linestyle=l_line_styles[algo],
            marker=l_markers[algo] if algo in l_markers else "o",
            # markersize=6,
            markevery=max(1, int(len(algo_data["mean"].index.values) / 15)),
            # fillstyle='none',
            **marker_args,
            label=algo if algo not in label_map else label_map[algo],
        )

        if median:
            ax.fill_between(
                algo_data["mean"].index.values,
                algo_data["sem"][0].values,
                algo_data["sem"][1].values,
                facecolor=colors[i],  # l_colors[algo] if algo in l_colors else f"C{i}",
                alpha=0.1,
                step="post"
            )
        else:
            ax.fill_between(
                algo_data["mean"].index.values,
                algo_data["mean"].values - algo_data["sem"].values,
                algo_data["mean"].values + algo_data["sem"].values,
                facecolor=colors[i],  # l_colors[algo] if algo in l_colors else f"C{i}",
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
    ax.set_ylabel("Normalized regret")  # \
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
