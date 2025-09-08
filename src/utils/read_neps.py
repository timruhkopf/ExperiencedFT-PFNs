import os
import fnmatch
import ast
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import fire
import yaml
from omegaconf import DictConfig, OmegaConf

from ifBO_icml2024.src.pfns_hpo.pfns_hpo.regret_plot import calculate_continuations


def find_files_recursive(root_dir, pattern):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(dirpath, filename))
    return matches
#
#
# def find_hydra_config_dir(filepath: str) -> Path:
#     # fixme: when copying from luis, the ".hydra" folder becomes "hydra",
#     #  causing this function to fail
#     path = Path(filepath).parent
#     while path != path.root:
#         hydra_dir = path / ".hydra"
#         if (hydra_dir / "config.yaml").exists() and (hydra_dir / "hydra.yaml").exists():
#             return hydra_dir
#         path = path.parent
#     raise FileNotFoundError(f"No .hydra/config.yaml found for {filepath}")

import re

def find_hydra_config_dir(filepath: str) -> Path:
    path = Path(filepath).parent
    pattern = re.compile(r"\.?hydra", re.IGNORECASE)
    while path != path.root:
        candidates = [d for d in path.iterdir() if d.is_dir() and pattern.match(d.name)]
        for hydra_dir in candidates:
            if (hydra_dir / "config.yaml").exists() and (hydra_dir / "hydra.yaml").exists():
                return hydra_dir
        path = path.parent
    raise FileNotFoundError(f"No hydra/config.yaml found for {filepath}")


def config_parser(config_path: Path, keys: List[str]) -> dict:
    new_config = {}
    with open(config_path / "hydra.yaml", 'r') as hydra_file:
        hydra_config = yaml.safe_load(hydra_file)
    hydra_config = DictConfig(hydra_config)
    if 'search_space' in hydra_config.hydra.sweeper:
        keys = keys + list(
            OmegaConf.select(hydra_config, 'hydra.sweeper.search_space.hyperparameters').keys())
    with open(config_path / "config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    config = DictConfig(config)
    for key in keys:
        new_config[key] = OmegaConf.select(config, key)
    return new_config


def parse_loss_config_file(filepath, hydra_config: Dict):
    with open(filepath, "r") as f:
        content = f.read()
    blocks = [block.strip() for block in content.split('-' * 79) if block.strip()]
    rows = []
    for block in blocks:
        lines = block.splitlines()
        loss = float(lines[0].split(":", 1)[1].strip())
        config_id = lines[1].split(":", 1)[1].strip()
        config_str = lines[2].split(":", 1)[1].strip()
        config = ast.literal_eval(config_str)
        row = {
            "loss": loss,
            "config_id": config_id,
            **config,
            **hydra_config,
            "filepath": filepath
        }
        rows.append(row)
    return pd.DataFrame(rows)


def parse_config_data(filepath, hydra_config: Dict,
                      continuations: bool = True,
            wallclock: bool = False,
            overhead: bool = False,
            strict: bool = False,
            analysis: bool = False

    ) -> pd.DataFrame:
    """Collects a single run for a benchmark-algorithm-seed combination."""
    filepath = Path(filepath)

    # (Parse the CSV file) -------------------------------
    assert filepath.exists(), f"{filepath} does not exist!"

    df = pd.read_csv(filepath, float_precision="round_trip")
    if analysis:
        try:
            df_analysis = pd.read_csv(
                filepath.parent.parent / "analysis_data.csv", float_precision="round_trip",
                index_col=0
            )
            df = pd.concat([df, df_analysis], axis=1)
        except:
            pass

    # sort by time
    # NOTE: key assumption - only single worker runs
    time_cols = ["result.info_dict.start_time", "result.info_dict.end_time"]
    df.sort_values(by=time_cols, inplace=True, ignore_index=True)

    # calculate continuations
    if continuations:
        if "arlind_root_directory" in str(filepath):
            df.loc[:, "result.info_dict.fidelity"] = 1
        else:
            calculate_continuations(df, inplace=True)

    # adding cumulative fidelities for the x-axis
    df.loc[:, "cumsum_fidelity"] = df["result.info_dict.fidelity"].cumsum()
    # make cumulative fidelity the index of the sorted dataframe
    # df.set_index("cumsum_fidelity")
    df = df.set_index(df.cumsum_fidelity.values)

    # adding a column with the incumbent trace of the `loss`
    df.loc[:, "inc_loss"] = np.minimum.accumulate(df["result.loss"].values)

    # adding benchmark costs
    df.loc[:, "benchmark_costs"] = (
            df.loc[:, "metadata.time_end"] - df.loc[:, "metadata.time_sampled"]
    )
    # adding sampling_times
    df.loc[:, "sampling_times"] = df.loc[:, "metadata.time_sampled"].diff(-1).fillna(0).abs()
    # adding overhead time
    df.loc[:, "overhead"] = (
            df.loc[:, "sampling_times"] - df.loc[:, "benchmark_costs"]
    ).clip(0, 1e24)

    # adding wallclock time
    df.loc[:, "wallclock_without_overhead"] = df.loc[:, "result.info_dict.cost"]  # .cumsum()
    # adding wallclock time with overhead
    df.loc[:, "wallclock_with_overhead"] = (
            df.loc[:, "wallclock_without_overhead"] + df.loc[:, "overhead"]  # .cumsum()
    )

    # summing up overhead time
    df.loc[:, "overhead"] = df.loc[:, "overhead"]  # .cumsum()

    if (wallclock or overhead) and continuations and "arlind_root_directory" not in str(filepath):
        fid_variable = "result.info_dict.fidelity"
        if wallclock and overhead:
            fid_variable = "wallclock_with_overhead"
        elif wallclock and not overhead:
            fid_variable = "wallclock_without_overhead"
        elif not wallclock and overhead:
            fid_variable = "overhead"

        # if fid_variable in ["wallclock_with_overhead", "wallclock_without_overhead", "overhead"]:
        if fid_variable in ["wallclock_without_overhead"]:
            # calculate continuations
            calculate_continuations(df, fid_var=fid_variable, inplace=True)
            # adding cumulative fidelities for the x-axis
            df.loc[:, "cumsum_fidelity"] = df[fid_variable].cumsum()
            # make cumulative fidelity the index of the sorted dataframe
            df = df.set_index(df.cumsum_fidelity.values)

    # (Add hydra config to all rows) -------------------------------
    for col, vals in hydra_config.items():
        df[col] = vals
    df["filepath"] = str(filepath)

    # (Parse variables from the filepath) -------------------------------
    # Regex pattern with named groups to extract the variables
    pattern = r'neps_root_directory_(?P<target_task>[^_]+)_(?P<fold>\d+)_(?P<split_seed>\d+)_(?P<seed>\d+)_(?P<allocation_seed>\d+)'

    # Extract variables to new columns
    df_vars = df['filepath'].str.extract(pattern)

    # Join extracted variables back to original dataframe if needed
    df = df.join(df_vars, rsuffix='_extracted')

    return df


def group_files_by_hydra(files: List[str]) -> Dict[Path, List[str]]:
    groups = {}
    for f in files:
        try:
            hydra_dir = find_hydra_config_dir(f)
            groups.setdefault(hydra_dir, []).append(f)
        except FileNotFoundError:
            continue
    return groups

def process_group(item, keys, file_pattern):
    hydra_dir, group_files = item
    hydra_config = config_parser(hydra_dir, keys.copy())
    if file_pattern == "all_losses_and_configs.txt":
        parse_func = partial(parse_loss_config_file, hydra_config=hydra_config)
    elif file_pattern == "config_data.csv":
        parse_func = partial(parse_config_data, hydra_config=hydra_config)
    dfs = [parse_func(f) for f in group_files]  # sequential inside group
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def parse_and_save(
        root_dir,
        keys: List[str],
        csv=None,
        file_pattern="all_losses_and_configs.txt",
        workers=4
):
    files = find_files_recursive(root_dir, file_pattern)
    if not files:
        print("No files found.")
        return

    grouped = group_files_by_hydra(files)

    if workers>1:
        with Pool(workers) as pool:
            # Map process_group over groups in parallel
            all_dfs = pool.map(partial(process_group, keys=keys, file_pattern=file_pattern), grouped.items())
    else:
        # non parallelized version for debugging
        all_dfs = [process_group(item, keys, file_pattern) for item in grouped.items()]

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if df.empty:
        print("No data parsed from files.")
        return

    if csv is not None:
        df.to_csv(csv, index=False)
        print(f"Saved {len(df)} rows to {csv}")

    return df


if __name__ == '__main__':
    fire.Fire(parse_and_save)

# python src/utils/read_neps.py /mnt/home/truhkopf/ExperiencedFT-PFNs/outputs/2025-05-13 --pattern all_losses_and_configs.txt --csv /mnt/home/truhkopf/ExperiencedFT-PFNs/neps_output1.csv --hydra_keys "[\"experiment_name\", \"benchmark.meta.name\", \"algorithm.name\", \"algorithm.surrogate_model.meta.name\"]"
