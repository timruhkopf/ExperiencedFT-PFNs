import time
import numpy as np

def run_optimization_loop(
    hpob_hdlr,
    optimizer,
    search_space_id = None, 
    dataset_id = None, 
    seed = None, 
    time_horizon = 50
):

    assert search_space_id!= None, "Provide a valid search space id. See documentatio for valid obptions."
    assert dataset_id!= None, "Provide a valid dataset_id. See documentation for valid options."
    assert seed!=None, "Provide a valid initialization. Valid options are: test0, test1, test2, test3, test4."
    try:
        X = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["X"])
        y = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["y"])
    except KeyError:
        print(hpob_hdlr.meta_test_data.keys())
        raise

    X = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["X"])
    y = np.array(hpob_hdlr.meta_test_data[search_space_id][dataset_id]["y"])
    y = hpob_hdlr.normalize(y)

    data_size = len(X)
    # indices of pending evaluations
    pending_evaluations = list(range(data_size))
    current_evaluations = []

    init_ids = hpob_hdlr.bo_initializations[search_space_id][dataset_id][seed]
    max_performance_history = []
    opt_time = []
    for i in range(len(init_ids)):
        start_time = time.time()
        idx = init_ids[i]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)
        max_performance_history.append(np.max(y[current_evaluations]))
        end_time = time.time()
        opt_time.append(end_time - start_time)

    # NOTE: change max to min
    for i in range(time_horizon - len(init_ids)):
        # take the acquistion values from the pending evaluations
        start_time = time.time()
        idx = optimizer.observe_and_suggest(
            X[current_evaluations], y[current_evaluations], X[pending_evaluations]
        )
        end_time = time.time()
        opt_time.append(end_time - start_time)
        idx = pending_evaluations[idx]
        pending_evaluations.remove(idx)
        current_evaluations.append(idx)
        max_performance_history.append(np.max(y[current_evaluations]))

    return max_performance_history, current_evaluations, opt_time
