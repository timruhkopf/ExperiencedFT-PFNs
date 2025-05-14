import os
import fnmatch
import ast
import re
from pathlib import Path
from typing import List

import pandas as pd
from multiprocessing import Pool
import fire
import yaml
from omegaconf import DictConfig, OmegaConf


#
# def parse_loss_config_file(filepath):
#     with open(filepath, "r") as f:
#         content = f.read()
#     blocks = [block.strip() for block in content.split('-' * 79) if block.strip()]
#     rows = []
#     for block in blocks:
#         lines = block.splitlines()
#         loss = float(lines[0].split(":", 1)[1].strip())
#         config_id = lines[1].split(":", 1)[1].strip()
#         config_str = lines[2].split(":", 1)[1].strip()
#         config = ast.literal_eval(config_str)
#         row = {
#             "loss": loss,
#             "config_id": config_id,
#             **config,
#             "filepath": filepath
#         }
#         rows.append(row)
#     return pd.DataFrame(rows)

def find_files_recursive(root_dir, pattern):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(dirpath, filename))
    return matches



#
# def extract_yaml_from_log(log_path, yaml_out_path=None):
#     """
#     # FIXME: Dirty Dirty quick fix, because neps didn't write out the .hydra file
#     Extracts a YAML block from a log file that contains specific `[INFO]: benchmark:`
#     entries. It processes the log file, identifies the relevant YAML block, and
#     either returns it as a string or saves it to an optional output file.
#
#     :param log_path: Path to the log file to be processed.
#     :type log_path: str
#     :param yaml_out_path: Optional path to save the extracted YAML block. If not provided,
#         the YAML block will not be written to a file.
#     :type yaml_out_path: str, optional
#     :return: Extracted YAML block as a string.
#     :rtype: str
#     :raises ValueError: If no YAML block starting with `[INFO]: benchmark:` is found
#         in the log file.
#     """
#     with open(log_path, 'r') as f:
#         lines = f.readlines()
#
#     # Pattern for log line (timestamp + src + [INFO]:)
#     log_line_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .*\[INFO\]:')
#
#     yaml_start = None
#     for i, line in enumerate(lines):
#         if '[INFO]: benchmark:' in line:
#             yaml_start = i
#             break
#
#     if yaml_start is None:
#         raise ValueError("Could not find the start of the YAML block.")
#
#     # The YAML starts after the '[INFO]: ' part
#     # Get the part after '[INFO]: ' in the start line, plus subsequent lines until the next log entry
#     yaml_lines = []
#     first_line = lines[yaml_start]
#     prefix = first_line.split('[INFO]:', 1)[1]
#     yaml_lines.append(prefix.lstrip())  # Remove leading spaces
#
#     for line in lines[yaml_start + 1:]:
#         if log_line_re.match(line):
#             break
#         yaml_lines.append(line)
#
#     yaml_block = ''.join(yaml_lines)
#
#     # Optionally save to file
#     if yaml_out_path:
#         with open(yaml_out_path, 'w') as f:
#             f.write(yaml_block)
#
#     return yaml_block


def find_hydra_config_dir(filepath: str) -> Path:
    """Finds the nearest parent directory containing .hydra/config.yaml"""
    path = Path(filepath).parent
    while path != path.root:
        hydra_dir = path / ".hydra"
        if (hydra_dir / "config.yaml").exists() and (hydra_dir / "hydra.yaml").exists():
            return hydra_dir
        path = path.parent
    raise FileNotFoundError(f"No .hydra/config.yaml found for {filepath}")


def config_parser(config_path: Path, keys: List[str]) -> dict:
    """Your existing config parser with minor adjustments"""
    new_config = {}

    with open(config_path / "hydra.yaml", 'r') as hydra_file:
        hydra_config = yaml.safe_load(hydra_file)

    hydra_config = DictConfig(hydra_config)
    if 'search_space' in hydra_config.hydra.sweeper:
        keys.extend(
            OmegaConf.select(hydra_config, 'hydra.sweeper.search_space.hyperparameters').keys())

    with open(config_path / "config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    config = DictConfig(config)
    for key in keys:
        new_config[key] = OmegaConf.select(config, key)

    return new_config




def parse_loss_config_file(filepath, hydra_keys: List[str]):
    # Get Hydra config first
    hydra_dir = find_hydra_config_dir(filepath)
    hydra_config = config_parser(hydra_dir, hydra_keys.copy())

    # Original parsing logic
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
            **hydra_config,  # Merge Hydra config
            "filepath": filepath
        }
        rows.append(row)

    return pd.DataFrame(rows)

def parse_and_save(
    root_dir,
    csv,
    hydra_keys: List[str],
    pattern="all_losses_and_configs.txt",
    workers=4
):
    """
    Recursively parses all matching files and saves the result as a CSV.

    Args:
        root_dir: Root directory to search for files.
        csv: Output CSV file path.
        pattern: Filename pattern to match (default: all_losses_and_configs.txt).
        workers: Number of parallel workers (default: 4).
    """
    files = find_files_recursive(root_dir, pattern)
    if not files:
        print("No files found.")
        return

        # Create partial function to pass hydra_keys to workers
    from functools import partial
    parse_func = partial(parse_loss_config_file, hydra_keys=hydra_keys)

    with Pool(workers) as pool:
        dfs = pool.map(parse_func, files)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df.to_csv(csv, index=False)
    print(f"Saved {len(df)} rows to {csv}")

if __name__ == '__main__':
    fire.Fire(parse_and_save)

    #python parse_configs.py ./mydir --csv output.csv --pattern "*.txt" --workers 8


# scp -r truhkopf@kisski01.cluster.uni-hannover.de:/ mnt/home/truhkopf/ExperiencedFT-PFNs/kisski_results/neps_summary.csv .


    #
    # # Usage
    # yaml_str = extract_yaml_from_log('/home/ruhkopf/PycharmProjects/ExperiencedFT-PFNs/kisski_results/full.log', yaml_out_path='recovered.yaml')
    # print(yaml_str)
