import argparse

import sys
sys.path.append("./HPO_B")  # Adjust the path to import from the parent directory
from hpob_handler import HPOBHandler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings = warnings
import torch
from run_hpob import run_optimization_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPOB benchmarks.")
    # SEARCH_SPACE_ID="4796 5527 5636 5859 5860 5891 5906 5965 5970 5971 6766 6767 6794 7607 7609 5889"
    parser.add_argument("--benchmark_name", type=str, default='HPO-B')
    parser.add_argument("--search_space_id", type=str, default='4796')
    parser.add_argument("--method", type=str, default='Random-Search')
    parser.add_argument("--time_horizon", type=int, default=60)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", type=str, default='cpu:0')
    parser.add_argument("--output", type=str, default='./results/')
    args = parser.parse_args()

    search_space_ids =  [args.search_space_id]
    time_horizon = args.time_horizon
    device = args.device
    seeds = ["test" + str(i)  for i in range(args.trials)]
    method_name = args.method
    output_dir = args.output
    benchmark_name = args.benchmark_name

    hpob_hdlr = HPOBHandler(root_dir="./HPO_B/hpob-data/", mode="v3-test")


    for search_space_id in search_space_ids:
        search_space_id = str(search_space_id) 
        dataset_ids = hpob_hdlr.get_datasets(search_space_id)
        for dataset_id in dataset_ids:
            results = []
            for seed in seeds:
                if method_name == 'Random-Search':
                    from methods.random_search import RandomSearch
                    method = RandomSearch()
                elif method_name == 'GP':
                    from optimizers.botorch import GaussianProcess
                    method = GaussianProcess("EI")
                elif method_name == 'PFNs4BO-HEBO':
                    import pfns4bo
                    from optimizers.pfns4bo import TransformerBOMethod
                    from pfns4bo.scripts.tune_input_warping import fit_input_warping
                    method = TransformerBOMethod(torch.load( pfns4bo.hebo_plus_model), device=device)
                elif method_name == "ourPFNs":
                    import pfns4bo
                    from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
                    from pfns4bo.scripts.tune_input_warping import fit_input_warping
                    from our_pfns4bo import ourTransformerBOMethod, get_meta_data
                    meta_data =  get_meta_data(benchmark_name, hpob_hdlr, search_space_id, dataset_id, dataset_ids,seeds,  seed)
                    method = ourTransformerBOMethod(torch.load( pfns4bo.hebo_plus_model), meta_data, device=device)
                elif method_name == 'HEBO':
                    from optimizers.hebo import HeboOptimizer
                    search_space_dim = hpob_hdlr.get_search_space_dim(search_space_id)
                    method = HeboOptimizer(search_space_dim, maximize=True)

                print(f"Evaluating method: {method_name} on dataset: {dataset_id} with search space id: {search_space_id} and seed: {seed}")
                performances, configuration_ids, opt_time = run_optimization_loop(hpob_hdlr, method, search_space_id = search_space_id, 
                                                    dataset_id = dataset_id,
                                                    seed = seed,
                                                    time_horizon = time_horizon )                             
                for ieration in range(time_horizon):
                    results.append({
                        "benchmark_name": benchmark_name,
                        "search_space_id": search_space_id,
                        "dataset_id": dataset_id,
                        "method": method_name,
                        "seed": int(''.join(filter(str.isdigit, seed))),
                        "iteration": ieration, 
                        "performance": performances[ieration],
                        "configuration": configuration_ids[ieration],
                        "duration": opt_time[ieration]
                    })
            df = pd.DataFrame(results)
            df.to_csv(output_dir + benchmark_name + "_" +  method_name  + "_" + search_space_id +  "_" + dataset_id + ".csv", index=False)