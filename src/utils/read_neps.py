import os
import fnmatch
import ast
import pandas as pd
from multiprocessing import Pool
import fire

def parse_loss_config_file(filepath):
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
            "filepath": filepath
        }
        rows.append(row)
    return pd.DataFrame(rows)

def find_files_recursive(root_dir, pattern):
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(dirpath, filename))
    return matches

def parse_and_save(
    root_dir,
    csv,
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
    with Pool(workers) as pool:
        dfs = pool.map(parse_loss_config_file, files)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df.to_csv(csv, index=False)
    print(f"Saved {len(df)} rows to {csv}")

if __name__ == '__main__':
    fire.Fire(parse_and_save)

    #python parse_configs.py ./mydir --csv output.csv --pattern "*.txt" --workers 8
