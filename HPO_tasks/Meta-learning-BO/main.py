import argparse
import json
import time
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings = warnings


import pandas as pd
import torch

from benchmarks.hpob import HPOBBench
from run_hpob import run_optimization_loop


if __name__ == "__main__":
    with open("benchmarks/HPO-B/hpob-data/meta-test-tasks-per-space.json", "r") as f:
        search_spaces = json.load(f)

    parser = argparse.ArgumentParser(description="Run HPOB benchmarks.")
    # SEARCH_SPACE_ID="4796 5527 5636 5859 5860 5891 5906 5965 5970 5971 6766 6767 6794 7607 7609 5889"
    parser.add_argument(
        "--search_space_id", choices=search_spaces.keys(), nargs="+", required=True
    )
    parser.add_argument("--dataset_id", nargs="+")

    parser.add_argument("--method", type=str, default="MALIBO")

    # TEST_SEED="test0 test1 test2 test3 test4"
    parser.add_argument("--test_seed", type=str, required=True)
    parser.add_argument("--continuous", action=argparse.BooleanOptionalAction)
    parser.add_argument("--evaluations", type=int, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--device", type=str, default='cpu:0')
    args = parser.parse_args()


    test_seed = args.test_seed # Valid options are: test0, test1, test2, test3, test4."
    

    # experiment_name = args.name if args.name else args.dataset
    root_dir = Path(args.output) if args.output else Path("./results/hpob/")
    is_continuous = args.continuous

    for search_space_id in args.search_space_id:
        if args.dataset_id is not None:
            datasets = args.dataset_id
        else:
            datasets = search_spaces[search_space_id]

        for dataset_id in datasets:
            benchmark = HPOBBench(
                search_space_id=search_space_id, dataset_id=dataset_id
            )

            test_seed = args.test_seed
            if args.test_seed == "all":
                seeds = [f"test{i}" for i in range(5)]
            else:
                seeds = [test_seed]

            for test_seed in seeds:
                seed_num = int(''.join(filter(str.isdigit, test_seed)))
                torch.manual_seed(seed_num)
                np.random.seed(seed_num)        

                if args.method == "MALIBO":
                    optimizer = MALIBO(benchmark.search_space, **classifier_config)
                    meta_dir = Path("./checkpoints_hpob") / "MALIBO" / f"{search_space_id}"
                    optimizer.save_dir = meta_dir
                    if meta_dir.exists():
                        optimizer = optimizer.load(meta_dir)
                        optimizer.classifier.meta_model.initialize(
                            0,
                            dtype=classifier_config["dtype"],
                            device=classifier_config["device"],
                        )
                    else:
                        meta_data, validation_data = benchmark.get_meta_data()
                        # MALIBO does not use the validation set
                        # For training MALIBO, each task needs learn a task embedding
                        # Testing on validation data without training on it is not possible
                        optimizer.meta_fit(meta_data, meta_dir=meta_dir, **train_config)

                elif args.method == "PFNs4BO":
                    import pfns4bo
                    from mpfns4bo.pfns4bo_opt import PFNs4BO
                    optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), device=args.device)

                elif args.method == "mPFNs4BO":
                    import pfns4bo
                    from mpfns4bo.mpfns4bo import MPFNs4BO
                    meta_data, validation_data = benchmark.get_meta_data()
                    optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, validation_data, device=args.device)
                else:
                    raise ValueError(f"Unknown method: {args.method}")
                # run BO loop
                regret, opt_time = run_optimization_loop(
                    benchmark=benchmark,
                    test_seed=test_seed,
                    optimizer=optimizer,
                    max_evaluations=args.evaluations,
                )

                output_dir = root_dir / test_seed / args.method / search_space_id
                output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"regret": regret}).to_csv(output_dir / f"{dataset_id}.csv")
                pd.DataFrame({"time": opt_time}).to_csv(
                    output_dir / f"{dataset_id}_opt_time.csv"
                )
                print(
                    f"Finished {args.method} on {search_space_id} for {dataset_id} with seed {test_seed}. Regret: {regret[-1]}, Time: {sum(opt_time)}"
                )
