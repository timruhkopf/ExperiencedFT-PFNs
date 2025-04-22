import numpy as np
import torch
from ifbo.utils import detokenize
from ifbo import Batch
from neps.search_spaces.search_space import pipeline_space_from_configspace
from neps.search_spaces.search_space import SearchSpace
import mfpbench


class LCBenchPrior:
    def __init__(self, task_id, data_path, seq_len=2000, n_fidelities=None, device="cpu"):
        benchmark = mfpbench.get(name="lcbench_tabular", task_id=task_id, datadir=data_path, preload=True, prior=None,
                                 remove_constants=True, seed=True, value_metric="val_balanced_accuracy", value_metric_test="test_balanced_accuracy")
        self.benchmark = benchmark
        self.space = benchmark.space
        self.dim_hyperparameters = len(benchmark.space)
        self.max_fidelities = benchmark.end
        self.ncurves = len(benchmark.configs)
        self.original_id = np.arange(self.ncurves)
        self.offset = min([int(_) for _ in benchmark.configs.keys()])

        self.n_fidelities = n_fidelities if n_fidelities is not None else \
            int(np.round(10 ** np.random.uniform(0, 3)))

        self.n_fidelities = min(self.n_fidelities, self.max_fidelities)

        self.seq_len = seq_len
        self.device = device

    def sample_dirichlet(self, alpha, eps=10**-9, single_eval_pos=500):
        if alpha is None:
            alpha = 10 ** np.random.uniform(-4, -1)

        weights = np.random.gamma(alpha, alpha, self.seq_len) + eps
        p = weights / np.sum(weights)

        # identify, which token belongs to which curve
        ids = np.arange(self.seq_len)
        all_levels = np.repeat(ids,
                               self.n_fidelities)  # since each curve has self.n_fidelities (fidelities) we
        # could observe, this is just a flat array for all curves after one another
        all_p = np.repeat(p, self.n_fidelities) / self.n_fidelities
        # The ordering vector is basically the mapping to the hp idx for each token
        # provided, that the token ordering is random and we are having a cutoff point where the
        # query starts, the subsequent for loop will do a "cumsum" over the occurences of the idx
        # to determine the length of that curve and where we are querying (past the query cutoff)
        ordering = np.random.choice(
            all_levels, p=all_p, size=self.seq_len, replace=False)

        cutoff_per_curve = np.zeros((self.seq_len,), dtype=int)
        epochs_per_curve = np.zeros((self.seq_len,), dtype=int)
        for i in range(self.seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        return cutoff_per_curve, epochs_per_curve, ordering

    # def sample_hyperparameters(self, ):
    #     return np.random.uniform(size=(self.seq_len, self.dim_hyperparameters))

    def _interpret_dirichlet_sample(self, ordering, epochs_per_curve, cutoff_per_curve, single_eval_pos):

        epoch = np.zeros(self.seq_len)
        id_curve = np.zeros(self.seq_len)
        original_id = np.arange(self.ncurves)

        curve_xs = []
        for cid in range(self.seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[:cutoff_per_curve[cid]] = np.arange(
                        1, cutoff_per_curve[cid]+1)
                # queries (if any)
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:
                    x_[cutoff_per_curve[cid]:] = np.random.choice(np.arange(
                        cutoff_per_curve[cid]+1, self.n_fidelities+1), size=epochs_per_curve[cid]-cutoff_per_curve[cid], replace=False)
                curve_xs.append(x_)
            else:
                curve_xs.append(None)

        # construct the batch data element
        curve_counters = torch.zeros(self.seq_len).type(torch.int64)
        for i in range(single_eval_pos):
            cid = ordering[i]
            id_curve[i] = cid + 1  # start from 1
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            curve_counters[cid] += 1

         # assign max fidelity to all curves in context
        unique_curves = np.unique(id_curve[:single_eval_pos])
        nbiud = len(id_curve[single_eval_pos:(
            single_eval_pos+len(unique_curves))])
        num_unique_curves = len(unique_curves)
        id_curve[single_eval_pos: single_eval_pos +
                 num_unique_curves] = unique_curves[:nbiud]
        end_pos = min(single_eval_pos + num_unique_curves, self.seq_len)
        epoch[single_eval_pos:end_pos] = self.max_fidelities

        task_data = []
        offset = min([int(_) for _ in self.benchmark.configs.keys()])
        for ordering, config_id, fidelity in zip(
            id_curve, original_id[id_curve.astype(int) - 1], epoch
        ):
            if ordering == 0:
                tmp = [0] * len(task_data[-1])
            else:
                _config_id = str(config_id + offset)
                tmp = []
                tmp = tmp + [ordering, fidelity]
                tmp = tmp + self._get_normalized_values(
                    config=self.benchmark.configs[_config_id], configuration_space=self.space
                )
                tmp = tmp + \
                    [self.benchmark.query(
                        config=_config_id, at=fidelity).error]
            task_data.append(tmp)

        task_data = np.array(task_data).astype(np.float32)
        config = torch.from_numpy(task_data[:, 2:-1])  # can sample right here?
        curve_val = torch.from_numpy(task_data[:, -1])

        # convert id_curve and epoch to torch tensors
        id_curve = torch.from_numpy(id_curve).to(torch.int64)

        epoch = torch.from_numpy(epoch).to(
            torch.int64) / self.n_fidelities  # normalize to [0,1]

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1)
        y = curve_val

        return x, y

    def sample_from_task(self, alpha, context_size):
        # sample exactly seq_len hps, but some may become inactive
        # hps = self.sample_hyperparameters()
        cutoff_per_curve, epochs_per_curve, ordering = self.sample_dirichlet(
            alpha=alpha,
            single_eval_pos=context_size,
        )
        x, y = self._interpret_dirichlet_sample(
            ordering=ordering,
            epochs_per_curve=epochs_per_curve,
            cutoff_per_curve=cutoff_per_curve,
            # hps=hps,
            single_eval_pos=context_size
        )
        return x, y

    def sample_batch(self, n_tasks, alphas, single_eval_pos):
        if alphas is None:
            alphas = [10 ** np.random.uniform(-4, -1) for _ in range(n_tasks)]
        if isinstance(alphas, float):
            alphas = [alphas] * n_tasks
        assert len(
            alphas) == n_tasks, "alphas must be a list of the same length as n_task"

        if single_eval_pos is None:
            single_eval_pos = int(np.round(self.seq_len / 2))
        if isinstance(single_eval_pos, int):
            single_eval_pos = [single_eval_pos] * n_tasks
        assert len(
            single_eval_pos) == n_tasks, "single_eval_pos must be a list of the same length as n_tasks"

        X = []
        Y = []
        for task, alpha, context_size in zip(range(n_tasks), alphas, single_eval_pos):
            x, y = self.sample_from_task(
                alpha=alpha, context_size=context_size)
            X.append(x)
            Y.append(y)

        X = torch.stack(X, dim=1).to(self.device).float()
        Y = torch.stack(Y, dim=1).to(self.device).float()

        return Batch(x=X, y=Y, target_y=Y.clone(), single_eval_pos=single_eval_pos)

    @staticmethod
    def _get_normalized_values(config, configuration_space):
        list_hp_names = configuration_space.get_hyperparameter_names()
        dict_values = config.as_dict()
        dict_values = dict((hp, dict_values[hp]) for hp in list_hp_names)

        neps_cfg = SearchSpace(
            **{
                k: v
                for k, v
                in pipeline_space_from_configspace(configuration_space).items()
                if k in list_hp_names
            }
        )

        neps_cfg.set_hyperparameters_from_dict(dict_values, defaults=False)
        res = [neps_cfg[k].normalized().value for k in list_hp_names]
        if any([res[_] is None for _ in range(len(res))]):
            print("WARNING: NaN values found in normalized values.")
            import pudb
            pudb.set_trace()

        return res


def detokenize_batch(batch: Batch):
    """
    Since detokenize only works for batch sizes of 1, we need to detokenize the batch
    by looping over the individual tasks. Here we additionally only collect the context
    :param batch:
    :return:
    """
    batches = [
        Batch(
            x=batch.x[:, i:i + 1, :],
            y=batch.y[:, i:i + 1],
            target_y=batch.target_y[:, i:i + 1],
            single_eval_pos=batch.single_eval_pos[i]
        ) for i in range(batch.x.shape[1])
    ]

    detokenized_tasks = [
        detokenize(b, b.single_eval_pos, device='cpu')
        for b in (batches)
    ]

    detokenized_contexts = [task[0] for task in detokenized_tasks]

    return detokenized_contexts


if __name__ == '__main__':

    lcbench_task_prior = LCBenchPrior(
        task_id="airlines",
        data_path="data/lcbench-tabular/",
        seq_len=2000,
        n_fidelities=None,
        device="cpu"
    )
    batch = lcbench_task_prior.sample_batch(
        n_tasks=5, alphas=0.1, single_eval_pos=1000)

    contexts = detokenize_batch(batch)

    print(contexts)
