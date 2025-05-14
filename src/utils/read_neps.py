import os
import fnmatch
import ast
from pathlib import Path
from typing import List, Dict
import pandas as pd
from multiprocessing import Pool
from functools import partial
import fire
import yaml
from omegaconf import DictConfig, OmegaConf

def find_files_recursive(root_dir, pattern):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(dirpath, filename))
    return matches

def find_hydra_config_dir(filepath: str) -> Path:
    path = Path(filepath).parent
    while path != path.root:
        hydra_dir = path / ".hydra"
        if (hydra_dir / "config.yaml").exists() and (hydra_dir / "hydra.yaml").exists():
            return hydra_dir
        path = path.parent
    raise FileNotFoundError(f"No .hydra/config.yaml found for {filepath}")

def config_parser(config_path: Path, keys: List[str]) -> dict:
    new_config = {}
    with open(config_path / "hydra.yaml", 'r') as hydra_file:
        hydra_config = yaml.safe_load(hydra_file)
    hydra_config = DictConfig(hydra_config)
    if 'search_space' in hydra_config.hydra.sweeper:
        keys = keys + list(OmegaConf.select(hydra_config, 'hydra.sweeper.search_space.hyperparameters').keys())
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

def group_files_by_hydra(files: List[str]) -> Dict[Path, List[str]]:
    groups = {}
    for f in files:
        try:
            hydra_dir = find_hydra_config_dir(f)
            groups.setdefault(hydra_dir, []).append(f)
        except FileNotFoundError:
            continue
    return groups

def parse_and_save(
    root_dir,
    csv,
    hydra_keys: List[str],
    pattern="all_losses_and_configs.txt",
    workers=4
):
    files = find_files_recursive(root_dir, pattern)
    if not files:
        print("No files found.")
        return

    grouped = group_files_by_hydra(files)
    all_dfs = []
    for hydra_dir, group_files in grouped.items():
        hydra_config = config_parser(hydra_dir, hydra_keys.copy())
        parse_func = partial(parse_loss_config_file, hydra_config=hydra_config)
        with Pool(workers) as pool:
            dfs = pool.map(parse_func, group_files)
        all_dfs.extend(dfs)
    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    df.to_csv(csv, index=False)
    print(f"Saved {len(df)} rows to {csv}")

if __name__ == '__main__':
    fire.Fire(parse_and_save)

# python src/utils/read_neps.py /mnt/home/truhkopf/ExperiencedFT-PFNs/outputs/2025-05-13 --pattern all_losses_and_configs.txt --csv /mnt/home/truhkopf/ExperiencedFT-PFNs/neps_output1.csv --hydra_keys "[\"experiment_name\", \"benchmark.meta.name\", \"algorithm.name\", \"algorithm.surrogate_model.meta.name\"]"
