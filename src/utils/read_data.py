import json
import os
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
from typing import Callable, List
from multiprocessing import Pool, cpu_count

from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


def config_parser(config_path, keys: List) -> dict:
    new_config = {}

    # parse the search space hyperparameter keys from the hydra config
    with open(config_path / "hydra.yaml", 'r') as hydra_file:
        hydra_config = yaml.safe_load(hydra_file)

    hydra_config = DictConfig(hydra_config)
    if 'search_space' in hydra_config.hydra.sweeper:
        keys.extend(OmegaConf.select(hydra_config, 'hydra.sweeper.search_space.hyperparameters').keys())

    with open(config_path / "config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    config = DictConfig(config)
    for key in keys:
        new_config[key] = OmegaConf.select(config, key)

    return new_config


def parse_timestamp(timestamp_str, pattern="%Y_%m_%d_%H%M%S_%f"):
    return datetime.strptime(timestamp_str, pattern)


def process_single_folder(
        args,
):
    file_path, config_parser, keys = args

    if file_path.suffix == '.jsonl':
        # If the file is a JSON file, read it as a DataFrame
        df = pd.read_json(file_path, lines=True)

    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)

    if df.empty:
        return None

    config_path = file_path.parent / '.hydra'
    config = config_parser(config_path, keys)

    for key, value in config.items():
        df[key] = value

    df['file_path'] = file_path.__str__()

    return df


def process_folders(
        root_dir,
        file_pattern,
        keys:List[str]=[],
        config_parser: Callable = config_parser,
        csv: str = None
):
    """
    Multiprocessing to collate the result data frames from multiple folders.

    :param root_dir:
    :param file_pattern: file to glob: e.g. '**/results.csv'
    :param keys: List of strings specifying the omegaconf key to parse.
    Assuming a hydra structure, there will be '/.hydra/config.yaml' files in
     each the associated parentfolder. This function will extract the
     keys from the config.yaml file and add them to the dataframe as items to
     identify the run.
    :param config_parser: parser function to extract the keys from the config.yaml file
    :param outputfile: optional output file to save the dataframe

    :return: pd.DataFrame

    :Note:
    To use this function from the command line, use the fire package.
    The keys argument is a bit tricky:

    # notice further that fire will inspect the output object (here a pd.DataFrame)
    # and we have access to the output object's attributes, functions and can pass values
    (see "- to_csv", which  makes the outputfile argument optional)

    :Fire example:
        --root_dir
        /home/ruhkopf/RegularizationGlitch/src/multirun/friedman/unknown_2025-02-13_13-28-50
        --file_pattern
        "**/results.csv"
        --keys
        "[\"dataset.meta.name\",\"budget\",\"model.cls.alpha\",\"fidelity_seed\"]"
        - to_csv
        /home/ruhkopf/RegularizationGlitch/src/multirun/friedman/unknown_2025-02-13_13-28-50/final_output.csv
    """
    all_dfs = []


    root = Path(root_dir)

    args = [(file, config_parser, keys) for file in root.rglob(file_pattern)]
    with Pool(cpu_count()) as p:
        dfs = p.map(process_single_folder, args)

    all_dfs.extend([df for df in dfs if df is not None])
    if bool(all_dfs):
        all_dfs = pd.concat(all_dfs, ignore_index=True)

        if csv:
            all_dfs.to_csv(csv, index=False)
    else:
        logger.info(f"No results found for {root_dir}")
        all_dfs = pd.DataFrame()

    return all_dfs


if __name__ == '__main__':
    import fire
    out = fire.Fire(process_folders)
    print(out)

    # df = process_single_folder(
    #     args=(
    #         Path(
    #             "/home/ruhkopf/RegularizationGlitch/src/multirun/friedman/unknown_2025-02-13_13-28-50/ridge_artificial_seed_42/subsetsize_0.30000000000000004/27/results.csv"),
    #         config_parser,
    #         ['dataset.meta.name']  # keys
    #     )
    # )
    #
    # df = process_folders(
    #     root_dir='/home/ruhkopf/RegularizationGlitch/multirun/mlp_sklearn_simple_clsf/999_fcb4ef8/',
    #     file_pattern="results.csv",
    #     keys=[
    #         'dataset.meta.name',
    #         'budget',
    #         'fidelity_seed'
    #     ]
    # )
    # print(df.head())
