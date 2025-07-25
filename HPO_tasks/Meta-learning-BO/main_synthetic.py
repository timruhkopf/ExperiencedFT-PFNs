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
import random

from benchmarks.SimpleSynth.simple_synth import run_optimization_loop, SimpleSynth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Synth benchmarks.")
    parser.add_argument("--method", type=str, default="MALIBO")
    parser.add_argument("--test_seed", type=str, default="all")
    parser.add_argument("--evaluations", type=int, default=50)
    parser.add_argument("--output", type=str, default="./results/synth/")
    parser.add_argument("--function", type=str, default="synth_1d")
    parser.add_argument("--device", type=str, default='cpu:0')
    args = parser.parse_args()

    root_dir = Path(args.output)
    test_seed = args.test_seed # Valid options are: test0, test1, test2, test3, test4."
    task_id = 8
    
    test_seed = args.test_seed
    if args.test_seed == "all":
        seeds = [f"test{i}" for i in range(0,10)]
    else:
        seeds = [test_seed]

    for test_seed in seeds:

        benchmark = SimpleSynth(seed=test_seed, start_task=task_id, initializations=4)

        seed_num = int(''.join(filter(str.isdigit, test_seed)))
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)
        random.seed(seed_num)

        if args.method == "MALIBO":
            from malibo.malibo import MALIBO
            classifier_config = {
                "num_layers": 5,
                "num_features": 50,
                "num_hidden_units": 64,
                "device": "cpu",
                "dtype": torch.float64,
            }
            train_config = {
                "num_epochs": 2048,
                "batch_size": 256,
            }
            optimizer = MALIBO(benchmark.search_space, **classifier_config)
            meta_dir = Path(args.output + "/checkpoints_synth") / "MALIBO" / f"{args.function}"
            optimizer.save_dir = meta_dir
            if meta_dir.exists():
                optimizer = optimizer.load(meta_dir)
                optimizer.classifier.meta_model.initialize(
                    0,
                    dtype=classifier_config["dtype"],
                    device=classifier_config["device"],
                )
            else:
                meta_data = benchmark.get_meta_data()
                # MALIBO does not use the validation set
                # For training MALIBO, each task needs learn a task embedding
                # Testing on validation data without training on it is not possible
                optimizer.meta_fit(meta_data, meta_dir=meta_dir, **train_config)

        elif args.method == "PFNs4BO-PI":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), device=args.device, acq_function_type='pi', apply_power_transform=True)

        elif args.method == "PFNs4BO-PI-o":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), device=args.device, acq_function_type='pi')

        elif args.method == "pPFNs4BO-decay-plus":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, apply_power_transform=False, mixing_strategy="my")


        elif args.method == "PFNs4BO-EI":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), device=args.device, acq_function_type='ei', apply_power_transform=True)

        elif args.method == "PFNs4BO-EI-IW":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            from pfns4bo.scripts.tune_input_warping import fit_input_warping
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), fit_encoder=fit_input_warping, device=args.device, acq_function_type='ei', apply_power_transform=True, input_power_transform=True)

        elif args.method == "PFNs4BO-PI-IW":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            from pfns4bo.scripts.tune_input_warping import fit_input_warping
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), fit_encoder=fit_input_warping, device=args.device, acq_function_type='pi', apply_power_transform=True, input_power_transform=True)


        elif args.method == "pPFNs4BO":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, apply_power_transform=True)

        elif args.method == "pPFNs4BO-cosine":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, apply_power_transform=True, tranformation_type='cosine')

        elif args.method == "pPFNs4BO-norm":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, apply_power_transform=True, tranformation_type='norm')

        elif args.method == "pPFNs4BO-linear":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, apply_power_transform=True, tranformation_type='linear')

        elif args.method == "mPFNs4BO":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device)

        elif args.method == "mPFNs4BO-incumbents":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, only_obs_incumbents=False)

        elif args.method == "mPFNs4BO-IW":
            import pfns4bo
            from mpfns4bo.mpfns4bo import MPFNs4BO
            meta_data = benchmark.get_meta_data()
            from pfns4bo.scripts.tune_input_warping import fit_input_warping
            optimizer = MPFNs4BO(torch.load( pfns4bo.hebo_plus_model), benchmark.search_space, meta_data, device=args.device, fit_encoder=fit_input_warping, apply_power_transform=True, input_power_transform=True)

        elif args.method == "PFNs4BO-PI-IW_meta_train":
            import pfns4bo
            from mpfns4bo.pfns4bo_opt import PFNs4BO
            from pfns4bo.scripts.tune_input_warping import fit_input_warping
            optimizer = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), fit_encoder=fit_input_warping, device=args.device, acq_function_type='pi', apply_power_transform=True, input_power_transform=True)



        else:
            raise ValueError(f"Unknown method: {args.method}")
        # run BO loop
        run_on_meta_train = False
        if "_meta_train" in args.method:
            run_on_meta_train = True

        regret, opt_time = run_optimization_loop(
            benchmark=benchmark,
            test_seed=test_seed,
            optimizer=optimizer,
            max_evaluations=args.evaluations,
            run_on_meta_train=run_on_meta_train
        )

        output_dir = root_dir / test_seed / args.method / args.function
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"regret": regret}).to_csv(output_dir / f"{task_id}.csv")
        pd.DataFrame({"time": opt_time}).to_csv(
            output_dir / f"{task_id}_opt_time.csv"
        )
        print(
            f"Finished {args.method} on {task_id} with seed {test_seed}. Regret: {regret[-1]}, Time: {sum(opt_time)}"
        )
