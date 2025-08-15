from typing import Optional, Union, List, Dict

from pathlib import Path
import numpy as np
import torch
from mfpbench import TabularBenchmark
from tqdm import tqdm

from src.dataset.synthetic_bm import SyntheticBenchmark
from ifbo.utils import detokenize
from ifbo import Batch
from neps.search_spaces.search_space import pipeline_space_from_configspace
from neps.search_spaces.search_space import SearchSpace
import mfpbench
from mfpbench.taskset_tabular.benchmark import TaskSetTabularBenchmark

import ConfigSpace as CS

LCBENCH_IDS = ['adult', 'airlines', 'albert', 'Amazon_employee_access', 'APSFailure',
               'Australian', 'bank-marketing', 'blood-transfusion-service-center', 'car',
               'christine', 'cnae-9', 'connect-4', 'covertype', 'credit-g', 'dionis', 'fabert',
               'Fashion-MNIST', 'helena', 'higgs', 'jannis', 'jasmine',
               'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'KDDCup09_appetency',
               'kr-vs-kp', 'mfeat-factors', 'MiniBooNE', 'nomao', 'numerai28.6', 'phoneme',
               'segment', 'shuttle', 'sylvine', 'vehicle', 'volkert']

PD1_IDS = [
    {"model": "wide_resnet", "dataset": "cifar10", "batch_size": 256},
    {"model": "wide_resnet", "dataset": "cifar10", "batch_size": 2048},
    {"model": "wide_resnet", "dataset": "cifar100", "batch_size": 256},
    {"model": "wide_resnet", "dataset": "svhn_no_extra", "batch_size": 256},
    {"model": "simple_cnn", "dataset": "fashion_mnist", "batch_size": 256},
    {"model": "simple_cnn", "dataset": "fashion_mnist", "batch_size": 2048},
    {"model": "simple_cnn", "dataset": "mnist", "batch_size": 256},
    {"model": "simple_cnn", "dataset": "mnist", "batch_size": 2048},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 256, "coarseness": 1},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 256, "coarseness": 2},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 256, "coarseness": 5},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 256, "coarseness": 10},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 512, "coarseness": 1},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 512, "coarseness": 2},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 512, "coarseness": 5},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 512, "coarseness": 10},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 1024, "coarseness": 1},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 1024, "coarseness": 2},
    {"model": "resnet", "dataset": "imagenet", "batch_size": 1024, "coarseness": 5},
    {"model": "transformer", "dataset": "lm1b", "batch_size": 2048},
    {"model": "transformer", "dataset": "uniref50", "batch_size": 128},
    {"model": "xformer_translate", "dataset": "translate_wmt", "batch_size": 64, "coarseness": 1},
    {"model": "xformer_translate", "dataset": "translate_wmt", "batch_size": 64, "coarseness": 2},
    {"model": "xformer_translate", "dataset": "translate_wmt", "batch_size": 64, "coarseness": 5},
    {"model": "xformer_translate", "dataset": "translate_wmt", "batch_size": 64, "coarseness": 10},
]

TASKSET_IDS = [
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64", "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64", "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_GRU128_bs128", "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128", "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128", "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
     "optimizer": "adam4p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64", "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64", "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_GRU128_bs128", "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128", "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128", "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
     "optimizer": "adam8p"},
    {"task_id": "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
     "optimizer": "adam8p"},
]


# for debugging purposes
def setup_debug_logger():
    """Create a dedicated debug logger that writes to a separate file"""
    import logging
    import os

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Configure logger
    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler('logs/debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add to handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handler to logger
    debug_logger.addHandler(file_handler)

    return debug_logger


# debug_logger = setup_debug_logger()


class MFBenchPrior(TabularBenchmark):

    def __init__(self,
                 name: str,
                 data_path = None,
                 seq_len=1000,
                 mfb_kwargs: Dict = None,
                 device="cpu",
                 **kwargs):
        self.name = name
        self.data_path = Path(data_path) if data_path is not None else None
        self.seq_len = seq_len
        self.mfb_kwargs = mfb_kwargs
        self.seq_len = seq_len
        self.device = device
        
        # for synthetic benchmarks
        # self.bnn_is_fixed = kwargs.get('bnn_is_fixed', True)
        self.n_layers = kwargs.get('n_layers', None)
        self.total_tasks = kwargs.get('total_tasks', 6)
        self.reset_kwargs = kwargs.get('reset_kwargs', {})
        self.seed = kwargs.get('seed', None)
        self.related_ratio = kwargs.get('related_ratio', 0.5)
        self.dim_hyperparameters = kwargs.get('dim_hyperparameters', 6)

    def collect_task_split(
            self,
            target_id: [int],
            train_ids: List[int],
            seed=None,
            bnn_seed=None,
    ):
        if self.name == "synthetic":
            self.target_benchmark = SyntheticBenchmark(value_metric="value",
                                                       cost_metric="fid_cost", seed=seed,
                                                       bnn_seed=bnn_seed, dim_hyperparameters=self.dim_hyperparameters)
            self.related_benchmarks = []
            
            for i in range(len(train_ids)):
                # if self.bnn_is_fixed:
                if i / len(train_ids) < self.related_ratio:
                    if self.seed is not None:
                        self.related_benchmarks.append(SyntheticBenchmark(value_metric="value", cost_metric="fid_cost", seed=seed + i + 1, dim_hyperparameters=self.dim_hyperparameters))
                    else:
                        self.related_benchmarks.append(SyntheticBenchmark(value_metric="value", cost_metric="fid_cost", dim_hyperparameters=self.dim_hyperparameters))
                else: # get a new Prior instance!
                    # related_task = self.target_benchmark.create_related_task(
                    #     n_layers=self.n_layers, dim_hyperparameters=self.dim_hyperparameters,
                    #     **self.reset_kwargs, same=False)
                    related_task= SyntheticBenchmark(value_metric="value",
                                                     cost_metric="fid_cost",
                                                     dim_hyperparameters=self.dim_hyperparameters, same=False)
                    self.related_benchmarks.append(related_task)

            # if self.bnn_is_fixed:
            #     assert all(self.target_benchmark.relation_prior.model is
            #                benchmark.relation_prior.model for benchmark in
            #                self.related_benchmarks)
                
        else:
            
            if not hasattr(self, 'train_ids'):
                # lazy load the train_ids (which takes time
                self.target_id = None
                self.train_ids = None

            if self.name == "lcbench_tabular":
                default_mfb_kwargs = {"name": self.name, "preload": True, "prior": None,
                                    "remove_constants": True, "seed": True,
                                    "value_metric": "val_balanced_accuracy",
                                    "value_metric_test": "test_balanced_accuracy"}
                target = {"task_id": LCBENCH_IDS[target_id]}

                if train_ids != self.train_ids:
                    self.related = [{"task_id": LCBENCH_IDS[task]}
                                    for task in train_ids]



            elif self.name == "pd1_tabular":

                if target_id >= len(PD1_IDS):
                    raise ValueError(
                        f"task_id {target_id} is out of bounds for PD1_IDS with size {len(PD1_IDS)}")
                if any(tid >= len(PD1_IDS) for tid in train_ids):
                    raise ValueError(
                        f"Some related_task_ids are out of bounds for PD1_IDS with size {len(PD1_IDS)}")
                default_mfb_kwargs = {"name": self.name,
                                    "preload": True, "prior": None, "seed": True}
                target = PD1_IDS[target_id]
                if train_ids != self.train_ids:
                    self.related = [PD1_IDS[task] for task in train_ids]
            elif self.name == "taskset_tabular":
                default_mfb_kwargs = {"name": self.name,
                                    "preload": True, "prior": None, "seed": True}
                target = TASKSET_IDS[target_id]
                if train_ids != self.train_ids:
                    self.related = [TASKSET_IDS[task] for task in train_ids]
            else:
                raise ValueError(
                    "name must be one of lcbench_tabular, pd1_tabular, or taskset")

            if self.mfb_kwargs is not None:
                default_mfb_kwargs.update(self.mfb_kwargs)
            mfb_kwargs = default_mfb_kwargs

            target_kwargs = mfb_kwargs.copy()
            target_kwargs.update(target)

            self.target_benchmark = mfpbench.get(
                datadir=self.data_path, **target_kwargs)

            self.related_benchmarks = []
            for related_task in self.related:
                related_kwargs = mfb_kwargs.copy()
                related_kwargs.update(related_task)
                self.related_benchmarks.append(
                    mfpbench.get(datadir=self.data_path, **related_kwargs))

            if self.name == "taskset_tabular":
                self.target_benchmark = self._process_taskset_mfpbench_with_step_0_prior(
                    benchmark=self.target_benchmark, drop_step_0=True
                )
                self.related_benchmarks = [
                    self._process_taskset_mfpbench_with_step_0_prior(
                        benchmark=benchmark, drop_step_0=True
                    )
                    for benchmark in self.related_benchmarks
                ]

            self.target_id = target_id
            self.train_ids = train_ids
        # self.name = self.name
        # self.fidelity_name = None
        # self.fidelity_range = None
        #
        # self.space: Union[SearchSpace, CS.ConfigurationSpace]
        # self.table =None

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        elif 'target_benchmark' in self.__dict__.keys():
            return self.__dict__['target_benchmark'].__dict__[name]

    def trajectory(self, config):  # -> what type?
        return self.target_benchmark.trajectory(config)

    def __len__(self):
        if self.name == "synthetic":
            return self.total_tasks
        else:
            return len(
                {
                    'lcbench_tabular': LCBENCH_IDS,
                    'pd1_tabular': PD1_IDS,
                    'taskset_tabular': TASKSET_IDS,
                }[self.name]
            )

    def sample_dirichlet(self, ncurves: int, max_fidelities: int, alpha: float = None,
                         eps: float = 10 ** -9,
                         single_eval_pos: int = 500):
        """
        Sample a Dirichlet distribution and compute token-to-curve associations with
        cutoff points and epochs per curve based on the generated probabilities.

        This function generates a Dirichlet distribution based on the provided or
        randomly determined concentration parameter (alpha) and assigns tokens
        to corresponding curves. It then calculates the number of epochs and
        cutoff values for each curve considering the position of the single evaluation.

        :param ncurves: The number of curves to sample from the Dirichlet distribution.
        :param alpha: Concentration parameter of the Dirichlet distribution. It
            determines the sparsity of the distribution. Defaults to a random value
            in the range [10^-4, 10^-1] if set to None.
        :param eps: A small constant added to avoid numerical instability, by default 10^-9.
        :param single_eval_pos: The number of tokens considered for single evaluation
            before the general distribution is applied, by default 500.
        :return: A tuple containing the cutoff points for each curve, the epochs per
            curve for assigning tokens, and the mapping of tokens to their respective curves.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """

        ok = False
        while not ok:
            if alpha is None:
                alpha = 10 ** np.random.uniform(-4, -1)

            n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
            n_levels = min(n_levels, max_fidelities)

            # weights = np.random.gamma(alpha, alpha, self.seq_len) + eps
            weights = np.random.gamma(alpha, alpha, min(1000, ncurves)) + eps
            p = weights / np.sum(weights)

            # identify which token belongs to which curve
            # ids = np.arange(self.seq_len)
            ids = np.arange(min(1000, ncurves))
            all_levels = np.repeat(ids, n_levels)
            # since each curve has self.n_fidelities (fidelities) we
            # could observe, this is just a flat array for all curves after one another
            all_p = np.repeat(p, n_levels) / n_levels

            if len(all_levels) > self.seq_len:
                ok = True

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

        return cutoff_per_curve, epochs_per_curve, ordering, n_levels

    def _interpret_dirichlet_sample(self, ordering, epochs_per_curve, cutoff_per_curve,
                                    single_eval_pos, benchmark, n_levels):
        """
        Processes Dirichlet sample data to generate task data and its corresponding
        tensor representations for model input and labels. The method uses the provided
        ordering and various attributes per curve to structure the observations and
        queries across fidelities.

        :param ordering: The ordering of curve execution as a list of curve indices.
        :type ordering: list[int]
        :param epochs_per_curve: A list indicating the number of epochs allocated
            to each curve.
        :type epochs_per_curve: list[int]
        :param cutoff_per_curve: A list indicating the cutoff point (number of epochs)
            for observations for each curve.
        :type cutoff_per_curve: list[int]
        :param single_eval_pos: The position in the sequence to evaluate a single curve
            before any others. This dictates the shape of the returned data.
        :type single_eval_pos: int

        :return: A tuple containing two tensors:
            - The first tensor contains the processed input data combining curve ID,
              epoch, configuration, and fidelity information.
            - The second tensor contains the labels (curve values) corresponding
              to the task data.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        epoch = np.zeros(self.seq_len)
        id_curve = np.zeros(self.seq_len)
        ncurves = len(benchmark.configs)
        original_id = np.arange(ncurves)

        max_fidelities = benchmark.end
        # debug_logger.debug(f"Max fidelities: {max_fidelities}")
        curve_xs = []
        for cid in range(self.seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[:cutoff_per_curve[cid]] = np.arange(
                        1, cutoff_per_curve[cid] + 1)
                # queries (if any)
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:
                    x_[cutoff_per_curve[cid]:] = np.random.choice(np.arange(
                        cutoff_per_curve[cid] + 1, n_levels + 1),
                        size=epochs_per_curve[cid] - cutoff_per_curve[cid], replace=False)
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
                single_eval_pos + len(unique_curves))])
        num_unique_curves = len(unique_curves)
        id_curve[single_eval_pos: single_eval_pos +
                                  num_unique_curves] = unique_curves[:nbiud]
        end_pos = min(single_eval_pos + num_unique_curves, self.seq_len)
        epoch[single_eval_pos:end_pos] = max_fidelities

        task_data = []
        offset = min([int(_) for _ in benchmark.configs.keys()])

        updated_space = False

        for ordering, config_id, fidelity in zip(
                id_curve, original_id[id_curve.astype(int) - 1], epoch
        ):
            if ordering == 0:
                tmp = [0] * len(task_data[-1])
            else:
                _config_id = str(config_id + offset)
                tmp = []
                tmp = tmp + [ordering, fidelity]

                config = benchmark.configs[_config_id]

                if self.name == "taskset_tabular":
                    optimizer_name = benchmark.name.split("-")[1].split("_")[0]

                    config = benchmark.configs[_config_id]

                    if optimizer_name == "adam4p":
                        if not updated_space:
                            updated_space = True
                            from ConfigSpace.hyperparameters import UniformFloatHyperparameter
                            benchmark.space.add_hyperparameters(
                                [
                                    UniformFloatHyperparameter(
                                        "l1",
                                        lower=1e-9,
                                        upper=10,
                                        log=True,
                                    ),
                                    UniformFloatHyperparameter(
                                        "l2",
                                        lower=1e-9,
                                        upper=10,
                                        log=True,
                                    ),
                                    UniformFloatHyperparameter(
                                        "linear_decay",
                                        lower=1e-8,
                                        upper=0.0001,
                                        log=True,
                                    ),
                                    UniformFloatHyperparameter(
                                        "exponential_decay",
                                        lower=1e-6,
                                        upper=1e-3,
                                        log=True,
                                    ),
                                ],
                            )

                        config = benchmark.configs[_config_id].as_dict()
                        default_values = {
                            "l1": 1e-7,
                            "l2": 1e-7,
                            "linear_decay": 1e-8,
                            "exponential_decay": 1e-6, }
                        config.update(default_values)

                    elif optimizer_name == "adam8p":
                        pass  # do nothing, we don't need to update the space

                    else:
                        raise ValueError(
                            f"Optimizer {optimizer_name} is currently not supported"
                        )

                tmp = tmp + self._get_normalized_values(
                    config=config, configuration_space=benchmark.space
                )
                if self.name == "synthetic":
                    tmp = tmp + \
                        [benchmark.query(config=config, at=fidelity).error]
                else:
                    tmp = tmp + \
                      [benchmark.query(
                          config=_config_id, at=fidelity).error]
            task_data.append(tmp)

        task_data = np.array(task_data).astype(np.float32)
        config = torch.from_numpy(task_data[:, 2:-1])  # can sample right here?
        curve_val = torch.from_numpy(task_data[:, -1])

        # convert id_curve and epoch to torch tensors
        id_curve = torch.from_numpy(id_curve).to(torch.int64)

        epoch = torch.from_numpy(epoch).to(
            torch.int64) / max_fidelities  # normalize to [0,1] same as https://github.com/automl/ifBO/blob/988e48ef4d9036d3670c32042906604321747823/src/section5.1/evaluate_pfn.py#L22

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1)
        y = curve_val

        return x, y

    def _collect_all_config_data(self, benchmark, n_fidelities):
        """
        Collects data for all configurations at all fidelities. in order to plot them.

        # JUST FOR DEBUGGING PLOTTING PURPOSES

        :param benchmark: Benchmark object containing configurations and query method.
        :param n_fidelities: Number of fidelities to evaluate per configuration.
        :return: Tuple of (input tensor, label tensor) containing all config-fidelity data.
        """
        task_data = []
        offset = min(int(k) for k in benchmark.configs.keys())

        for config_key in benchmark.configs:
            config = benchmark.configs[config_key]
            config_id = int(config_key) - offset  # Adjust for offset
            normalized_config = self._get_normalized_values(config, benchmark.space)

            # Query all fidelities from 1 to n_fidelities
            for fidelity in range(1, n_fidelities + 1):
                error = benchmark.query(config=config_key, at=fidelity).error
                task_data.append([
                    config_id + 1,  # Match original 1-based indexing
                    fidelity,
                    *normalized_config,
                    error
                ])

        # Convert to tensors
        task_data = np.array(task_data, dtype=np.float32)
        config_ids = torch.from_numpy(task_data[:, 0]).long()
        fidelities = torch.from_numpy(task_data[:, 1]).float() / n_fidelities  # Normalize
        config_features = torch.from_numpy(task_data[:, 2:-1]).float()
        labels = torch.from_numpy(task_data[:, -1]).float()

        x = torch.cat([
            torch.stack([config_ids, fidelities], dim=1),
            config_features
        ], dim=1)

        self._plot_data(x, labels, len(benchmark.configs), n_fidelities)

        return x, labels

    @staticmethod
    def _plot_data(x, y, num_configs, n_fidelities):

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for config_id in range(1, num_configs + 1):
            mask = x[:, 0] == config_id
            fidelities = x[mask, 1] * n_fidelities  # denormalize for plotting
            errors = y[mask]
            plt.plot(fidelities, errors, marker='o', label=f'Config {config_id}')

        plt.xlabel('Fidelity')
        plt.ylabel('Error')
        plt.title('Error vs Fidelity for each Configuration')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sample_from_task(self, alpha, context_size, benchmark):
        """
        Samples a task from the distribution specified by the Dirichlet process
        using provided context size and alpha value.

        This function generates samples while respecting the Dirichlet distribution.
        It determines cutoff points, epochs per data curve, and their ordering.
        The results are interpreted and returned in the form of input-output pairs.

        :param alpha: A float specifying the concentration parameter for the Dirichlet
            distribution, which influences how uneven the distribution of samples is.
        :param context_size: An integer that denotes the size of the context that is
            used for sampling and evaluation.
        :return: A tuple `(x, y)` where `x` represents the input data
            and `y` represents the corresponding output data.
        """

        # self._collect_all_config_data(benchmark, benchmark.end)

        ncurves = len(benchmark.configs)

        max_fidelities = benchmark.end

        cutoff_per_curve, epochs_per_curve, ordering, n_levels = self.sample_dirichlet(
            ncurves=ncurves,
            max_fidelities=max_fidelities,
            alpha=alpha,
            single_eval_pos=context_size,
        )

        x, y = self._interpret_dirichlet_sample(
            ordering=ordering,
            epochs_per_curve=epochs_per_curve,
            cutoff_per_curve=cutoff_per_curve,
            single_eval_pos=context_size,
            benchmark=benchmark,
            n_levels=n_levels,
        )

        return x, y

    def sample_batch(self, alphas: Optional[Union[List[float], float]] = None,
                     single_eval_pos=None, **kwargs, ):
        """
        Generates a batch of data sampled from multiple tasks.

        This method creates a batch of data by sampling from multiple tasks using
        specified alpha values (scale parameter for task variability) and evaluation
        positions within the sequence.

        :param n_tasks: The number of tasks to sample data for.
        :param alphas: A list or single float value representing scale parameters
            (alpha values) for the tasks. If None, random alpha values are
            generated. If a single float is provided, it is applied to all tasks.
        :param single_eval_pos: A list or single integer specifying the evaluation
            position(s) within the sequence for the tasks. Defaults to the middle
            of the sequence for all tasks if None. If a single integer is provided,
            it is applied to all tasks.
        :type single_eval_pos: Optional[Union[List[int], int]]
        :return: A Batch object containing:
            - X: The input data stacked across tasks and moved to the specified
              device as a FloatTensor.
            - Y: The output/target data stacked across tasks and moved to the
              specified device as a FloatTensor.
            - target_y: A copy of the target (Y) data for potential use in model
              evaluation.
            - single_eval_pos: The list of evaluation positions used for each task.
        :rtype: Batch
        """
        if self.name == "synthetic" and self.data_path is not None and (self.data_path /
                'config.json').exists():
            # load the bnn config and weights to have the same target benchmark task as the related tasks
            from ifbo.priors.ftpfn_prior import MLP
            import json
            self.target_benchmark.relation_prior.model = MLP(2,3)
            self.target_benchmark.relation_prior.model.to(self.device)

            with open(self.data_path / 'config.json', 'r') as f:
                config = json.load(f)

            self.target_benchmark.relation_prior.model.reset_from_signature(config)
            self.target_benchmark.relation_prior.model.load_state_dict(
                torch.load(self.data_path / 'benchmark_model.pt',
                           map_location=self.device,
                           weights_only=True)
            )
            assert self.target_benchmark.relation_prior.model is self.related_benchmarks[0].relation_prior.model
            if self.related_ratio != 1.0:
                assert not self.related_benchmarks[0].relation_prior.model is \
                           self.related_benchmarks[1].relation_prior.model
            return torch.load(self.data_path / 'batch.pt', map_location=self.device,
                              weights_only=False)

        benchmarks = self.related_benchmarks
        print(len(benchmarks), "benchmarks")
        print(alphas)
        if alphas is None:
            alphas = [10 ** np.random.uniform(-4, -1) for _ in range(len(benchmarks))]
        if isinstance(alphas, float):
            alphas = [alphas] * len(benchmarks)
        print(len(alphas), "alphas")
        assert len(alphas) == len(benchmarks), \
            "alphas must be a list of the same length as n_task"

        if single_eval_pos is None:
            single_eval_pos = int(np.round(self.seq_len / 2))
        if isinstance(single_eval_pos, int):
            single_eval_pos = [single_eval_pos] * len(benchmarks)
        assert len(single_eval_pos) == len(benchmarks), \
            ("single_eval_pos must be a list of the same length as n_related_tasks")

        X = []
        Y = []
        for task, alpha, context_size in tqdm(zip(
                benchmarks,
                alphas,
                single_eval_pos,
        ), desc="Sampling related task's budget allocations:", total=len(benchmarks)):
            x, y = self.sample_from_task(
                alpha=alpha, context_size=context_size,
                benchmark=task,
            )
            X.append(x)
            Y.append(y)

        X = torch.stack(X, dim=1).to(self.device).float()
        Y = torch.stack(Y, dim=1).to(self.device).float()
        if self.name == "synthetic" and self.data_path is not None:
            self.data_path.mkdir(parents=True, exist_ok=True)
            import json
            with open(self.data_path / 'config.json', 'w') as f:
                json.dump(self.target_benchmark.relation_prior.model.parameter_signature(), f)
            torch.save(Batch(x=X, y=Y, target_y=Y.clone(), single_eval_pos=single_eval_pos),
                       self.data_path / 'batch.pt')
            torch.save(self.target_benchmark.relation_prior.model.state_dict(), self.data_path /
                       'benchmark_model.pt')

        return Batch(x=X, y=Y, target_y=Y.clone(), single_eval_pos=single_eval_pos)

    @staticmethod
    def _get_normalized_values(config, configuration_space):
        """
        Normalize configuration values using a given configuration space.

        This function extracts the hyperparameter names from the configuration
        space and retrieves their corresponding values from the given
        configuration. It normalizes these values using the NEPS search space
        framework and returns the normalized results.

        :param config: Configuration object containing the parameter values to
            be normalized based on the configuration space.
        :type config: dict
        :param configuration_space: Configuration space defining the
            hyperparameters and their range in which the normalization is
            applied.
        :type configuration_space: ConfigurationSpace
        :return: A list of normalized values corresponding to the
            hyperparameters within the configuration space.
        :rtype: list[float]
        """

        list_hp_names = configuration_space.get_hyperparameter_names()
        if isinstance(config, dict):
            dict_values = config
        else:
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

    # https://github.com/automl/ifBO/blob/icml-2024/src/pfns_hpo/pfns_hpo/run.py

    @staticmethod
    def _process_taskset_mfpbench_with_step_0_prior(
            benchmark: TaskSetTabularBenchmark,
            loss_name_at_step_0_to_normalize_with: str = "valid1_loss",
            metrics_to_normalize_and_clamp: tuple[str, ...] = (
                    "train_loss",
                    "valid1_loss",
                    "valid2_loss",
                    "test_loss",
            ),
            drop_step_0: bool = True,
    ) -> TaskSetTabularBenchmark:
        """Normalize metrics of benchmark with the step 0 median huerisitc.

        To handle the case of unknown upper bounds, we need to normalize the results
        returned to algorithms in a (0, 1) range. We do this through clipping to
        some hueristic upper bound and then normalizing w.r.t. to this upper bound.

        In practice, a practioner could provide this upper bound value from the prior,
        either from rapid experimentation or previous experimental trials. They would
        then normalize the results before feeding it the hpo algorithm. For convenience
        sake, we apply this hueristic across the table using this prior knowledge.

        To generalize across tables, where peeking at the maximum loss value could be
        considered leaking information, we apply a heuristic that we believe matches
        the above intuitions, namely we use the median loss at step 0 as the upper
        bound. This corresponds to the median loss of the initial random configuration
        and is a reasonable heuristic for the upper bound. Notably, this does not look
        deep into the table to perform normalizations.

        Args:
            benchmark: The benchmark to normalize
            loss_name_at_step_0_to_normalize_with: The loss column from which
                to apply the hueristic and get the upper bound. By default,
                this is the validation loss used in the benchmark.
            metrics_to_normalize_and_clamp: The metric columns to normalize and clamp
                according to the upper bound. By default, this applies to all
                loss columns.
            drop_step_0: Whether to drop the step 0 values after normalization. By default,
                this will drop the step 0 values. These are random initializations and would
                generally not be reported to a HPO algorithm.
        """
        table = benchmark.table

        # Make sure that the table has a step column first
        if "step" not in table.columns:
            raise ValueError("Benchmark does not have 'step' in columns")

        # Make sure benchmarks contain the random initializations we use for the heuristic
        values = table[table["step"] == 0]
        assert values.index.is_unique, f"Step 0 not unique across configs?\n{values.index}"

        # Make sure that the loss column we are using to normalize with is non-negative
        # and corresponds to a regular log-loss, where the known theoretical bound is 0.
        # This is so the min-max normalization is well-defined and we can rely on the
        # theoretical bound to be 0 for normalization.
        if table[loss_name_at_step_0_to_normalize_with].min() < 0:
            raise ValueError(
                f"Benchmark has negative loss in '{loss_name_at_step_0_to_normalize_with}',"
                " normalization can not be applied naively",
            )

        # Get the median loss at step 0 as the hueuristic upper bound
        median_loss_to_use_as_prior = values[loss_name_at_step_0_to_normalize_with].median(
        )

        # Include the normalizing bound in the table
        # NOTE: This causes a crash with neps and isn't essneital
        # table[
        # f"normalizing_bound_from_{loss_name_at_step_0_to_normalize_with}"
        # ] = median_loss_to_use_as_prior

        for metric in metrics_to_normalize_and_clamp:
            # Make sure to store the corresponding non-normalized metric
            # NOTE: This causes a crash with neps and isn't essneital
            # table[f"{metric}_unnormalized"] = table[metric].copy()

            # Clip according to the median loss at step 0
            table[metric] = table[metric].clip(
                lower=0, upper=median_loss_to_use_as_prior)

            # And then normalize w.r.t. to this upper bound
            table[metric] = table[metric] / median_loss_to_use_as_prior

        if drop_step_0:
            # Select all rows that are not step 0
            table = table[table["step"] != 0]

            # Decrease the epoch number by one for uniformity
            table = table.reset_index()
            table["epoch"] = table["epoch"] - 1
            table = table.set_index(["id", "epoch"]).sort_index()

            # Decrease the fidelity range by 1
            lower, upper, step = benchmark.fidelity_range
            benchmark.fidelity_range = (lower, upper - 1, step)

        benchmark.table = table

        if benchmark.table.isna().any().any():
            print(benchmark.table.isna().any())
            raise ValueError("There should not be an na's left")

        return benchmark


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
        detokenize(b, b.single_eval_pos, device=b.x.device)
        for b in (batches)
    ]

    detokenized_contexts = [task[0] for task in detokenized_tasks]

    return detokenized_contexts


if __name__ == '__main__':
    from pathlib import Path

    lcbench_task_prior = MFBenchPrior(
        name="lcbench_tabular",
        data_path=Path(__file__).parents[2] / "data/lcbench-tabular/",
        seq_len=1000,
        device="cpu"
    )
    lcbench_task_prior.collect_task_split(
        target_id=0,
        train_ids=[1, 2, 3, 4, 5, 6, 7, 8]
    )

    pd1bench_task_prior = MFBenchPrior(
        name="pd1_tabular",
        data_path=Path(__file__).parents[2] / "data/pd1-tabular/",
        seq_len=1000,
        device="cpu"
    )
    pd1bench_task_prior.collect_task_split(
        target_id=14,
        train_ids=[1, 2, 3]
    )

    taskset_task_prior = MFBenchPrior(
        name="taskset_tabular",
        data_path=Path(__file__).parents[2] / "data/taskset-tabular/",
        seq_len=1000,
        device="cpu"
    )
    taskset_task_prior.collect_task_split(
        target_id=8,
        train_ids=[21, 5, 2, 12, 15]
    )

    batch = taskset_task_prior.sample_batch(alphas=0.1, single_eval_pos=500)

    contexts = detokenize_batch(batch)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from itertools import count

    # Convert to DataFrame with curve identifiers
    data = []
    curve_counter = count(1)

    for ctx_idx, curves in enumerate(contexts):
        for curve in curves:
            curve_id = f"Curve {next(curve_counter)}"
            for t_val, y_val in zip(curve.t, curve.y):
                data.append({
                    'Context': f'Context {ctx_idx}',
                    'Curve': curve_id,
                    't': t_val,
                    'y': y_val
                })

    df = pd.DataFrame(data)

    # Create facet grid with hue differentiation
    g = sns.FacetGrid(df, col='Context', hue='Curve',
                      col_wrap=4, height=4, aspect=1.2,
                      palette='viridis')

    # Map individual lines for each curve
    g.map(plt.plot, 't', 'y', linewidth=1.5)

    # Configure plot aesthetics
    g.set(ylim=(0, 1), xlim=(0, 1))
    g.set_axis_labels("step (t)", "performance (y)")
    # g.add_legend(title='Curve ID')
    g.fig.suptitle("Performance Curves by Context", y=1.05)

    plt.tight_layout()
    plt.show()

    print(contexts)
