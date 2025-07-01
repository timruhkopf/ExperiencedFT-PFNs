import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings = warnings
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic functions.")
    parser.add_argument("--function_name", type=str, default='Branin')
    parser.add_argument("--method", type=str, default='Random-Search')
    parser.add_argument("--time_horizon", type=int, default=105)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--device", type=str, default='cpu:0')
    parser.add_argument("--output", type=str, default='./results/')
    args = parser.parse_args()

    time_horizon = args.time_horizon
    device = args.device
    seeds = ["test" + str(i)  for i in range(args.trials)]
    method_name = args.method
    output_dir = args.output
    function_name = args.function_name

    from utils import ScaledFunctionWrapper
    if function_name == 'Branin':
        from botorch.test_functions.synthetic import Branin
        func = Branin()

    elif function_name == 'DropWave':
        from botorch.test_functions.synthetic import DropWave
        func = DropWave()

    elif function_name == 'Levy':
        from botorch.test_functions.synthetic import Levy
        func = Levy()

    elif function_name == 'Ackley':
        from botorch.test_functions.synthetic import Ackley
        func = Ackley()

    elif function_name == 'Rastrigin':   
        from botorch.test_functions.synthetic import Rastrigin
        func = Rastrigin()

    elif function_name == 'Rosenbrock':  
        from botorch.test_functions.synthetic import Rosenbrock
        func = Rosenbrock()



    function_ids ={}
    function_ids["0"]=  ScaledFunctionWrapper(func, y_transform=None, X_transforms=None)
    function_ids["1"]=  ScaledFunctionWrapper(func, y_transform=np.log1p, X_transforms=None)

    X_transforms = [lambda x: 1-x for i in range(func.dim)]
    function_ids["2"]=  ScaledFunctionWrapper(func, y_transform=None, X_transforms=X_transforms)

    X_transforms = [lambda x: x if i%2 else 1-x for i in range(func.dim)]
    function_ids["3"]=  ScaledFunctionWrapper(func, y_transform=None, X_transforms=X_transforms)

    X_transforms = [lambda x: 0.5*x+0.5 if i%2 else 0.5-0.5*x for i in range(func.dim)]
    function_ids["4"]=  ScaledFunctionWrapper(func, y_transform=None, X_transforms=X_transforms)

    y_transform = lambda y: y + y*2 - 0.1* y*3
    function_ids["5"]=  ScaledFunctionWrapper(func, y_transform=y_transform, X_transforms=None)

    results = []
    for transform_id, func in function_ids.items():
        for seed in seeds:
            seed_num = int(''.join(filter(str.isdigit, seed)))
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

            if method_name == 'Random-Search':
                from simple_random_search import RandomSearch
                method  = RandomSearch(bounds=func.bounds)
            elif method_name == 'GP-UCB':
                from simple_bayes_opt import BayesianOptimizer
                method  = BayesianOptimizer(bounds=func.bounds)
            elif method_name == 'PFNs4BO-HEBO':
                from simple_pfns4bo import PFNs4BO
                import pfns4bo
                from pfns4bo.scripts.tune_input_warping import fit_input_warping
                method  = PFNs4BO(torch.load( pfns4bo.hebo_plus_model), bounds=func.bounds)

            elif method_name == "ourPFNs":
                    meta_data_lists = [
                        ("Branin", "0", "Random-Search"),
                        ("Branin", "1", "Random-Search"),
                        ("Branin", "2", "Random-Search"),
                        ("Branin", "3", "Random-Search"),
                        ("Branin", "4", "Random-Search"),
                        ("Branin", "5", "Random-Search"),
                    ]

                    import pfns4bo
                    from pfns4bo.scripts.tune_input_warping import fit_input_warping
                    from our_pfns4bo import ourPFNs4BO, get_meta_data_of_method_name
                    meta_data =  get_meta_data_of_method_name(meta_data_lists, seeds,  seed )
                    method = ourPFNs4BO(torch.load( pfns4bo.hebo_plus_model), meta_data, bounds=func.bounds, device=device)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            print(f"Evaluating method: {method_name} on function: {function_name, transform_id} and seed: {seed}")
            
            for ieration in range(time_horizon):
                x = method.suggest()
                y = func(x)
                method.observe(x, y)

                if  method_name == "ourPFNs":
                    results.append({
                        "function_name": function_name,
                        "transform_id": transform_id,
                        "method": method_name,
                        "seed": int(''.join(filter(str.isdigit, seed))),
                        "iteration": ieration, 
                        "y": float(y),
                        "x": x ,
                        "reliability_scores" :method.reliability_scores,
                    })
                else:
                    results.append({
                        "function_name": function_name,
                        "transform_id": transform_id,
                        "method": method_name,
                        "seed": int(''.join(filter(str.isdigit, seed))),
                        "iteration": ieration, 
                        "y": float(y),
                        "x": x 
                    })
    df = pd.DataFrame(results)
    df.to_csv(output_dir + function_name + "_" +  method_name   + ".csv", index=False)