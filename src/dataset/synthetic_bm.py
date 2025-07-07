from ConfigSpace import ConfigurationSpace
from typing import Any, ClassVar, Generic, Mapping, TypeVar
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ifbo.priors.ftpfn_prior import DatasetPrior
from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.metric import Metric
from mfpbench.result import Result
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing_extensions import override
import torch
import copy


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class SynstheticConfig(Config):
    """Configuration for the synthetic benchmark."""

    # Define the hyperparameters for the synthetic benchmark
    X_0: float
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float


@dataclass(frozen=True)
class SyntheticBenchmarkResult(Result[SynstheticConfig, int]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "value": Metric(minimize=False, bounds=(0.0, 1.0)),
        "fid_cost": Metric(minimize=True, bounds=(0.05, 1.0)),
    }
    default_value_metric: ClassVar[str] = "value"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[str] = "fid_cost"

    value: Metric.Value
    fid_cost: Metric.Value


C = TypeVar("C", bound=Config)
R = TypeVar("R", bound=Result)


class BNNManager:
    _instance = None
    _initialized = False

    @classmethod
    def get_instance(cls, dim_hyperparameters, n_curve_param):
        if cls._instance is None:
            cls._instance = DatasetPrior(dim_hyperparameters, n_curve_param)
            cls._instance.new_dataset()
            cls._initialized = True
        return cls._instance


class SyntheticBenchmark(Benchmark):

    config_type: type[C] = SynstheticConfig
    result_type: type[R] = SyntheticBenchmarkResult

    def __init__(
        self,
        *,
        seed: int | None = None,
        prior: str | Path | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        cost_metric: str | None = None,
        dim_hyperparameters: int  = 6,
        max_fidelities: int = 50,
    ):
        self.n_configs = 500 # Number of configurations to sample
        self.max_fidelities = max_fidelities  # Maximum number of epochs

        # sample the dimensionality (i.e. hp space dim)
        self.dim_hyperparameters = dim_hyperparameters
        self.sampling_seed = seed
        
        # set up the BNN mapping from the hyperparameter space to the learning curves
        n_curve_param = 23  # Number of parameters for the learning curve basis and their weights
        self.relation_prior = BNNManager.get_instance(
            dim_hyperparameters=self.dim_hyperparameters,
            n_curve_param=n_curve_param,
        )

        name = (
            f"SyntheticBenchmark_{self.dim_hyperparameters}D"
        )

        # get unit cube hyperparameter space
        space = ConfigurationSpace(name=name, seed=seed)
        space.add_hyperparameter(
            [
                UniformFloatHyperparameter(
                    name=f"X_{i}",
                    lower=0.0,
                    upper=1.0,
                ) for i in range(self.dim_hyperparameters)
            ],
        )

        super().__init__(
            name=name,
            config_type=self.config_type,
            result_type=self.result_type,
            fidelity_name="epochs",
            fidelity_range=(0, self.max_fidelities - 1, 1),
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
            cost_metric=cost_metric,
        )

    @override
    def _objective_function(
        self, config: Mapping[str, Any], *, at: int,
    ) -> dict[str, float]:
        # convert config to np array
        config = np.array([config[f"X_{i}"]
                           for i in range(self.dim_hyperparameters)])

        curves = self.relation_prior.curves_for_configs(config[np.newaxis, :])
        # what should be the first argument?
        return {"value": curves(np.array([at/self.max_fidelities]), 0)[0], "fid_cost": self._fidelity_cost(at)}
    
    def _fidelity_cost(self, at: int) -> float:
        return 0.05 + (1 - 0.05) * (at / self.fidelity_range[1]) ** 2
        

    def create_related_task(self, n_layers, **reset_kwargs):
        if n_layers is None:
            AssertionError(
                "n_layers must be specified to create a related task.")
        new_benchmark = SyntheticBenchmark(
            seed=self.seed,
            prior=self.prior,
            perturb_prior=self.perturb_prior,
            value_metric=self.value_metric,
        )

        original_bnn = self.relation_prior.model
        new_bnn = copy.deepcopy(original_bnn)
        new_benchmark.relation_prior.model = new_bnn
        new_benchmark._sample_related_task(n_layers=n_layers, **reset_kwargs)
        return new_benchmark

    def _sample_related_task(
            self,
            n_layers: int,
            init_std=None,
            sparseness=None,
            pre_activation_noise=0,
            output_noise=0
    ):
        """
        Sample a related task. Here we make use of the fact, that the BNN (MLP) is instantiated
        with some depth, width and weights. To establish a new task, that has a similar mapping
        from hp to lc, we can resample some weights of the BNN. Specifically, we can determine
        the degree of resampling by the number of layers that are resampled, starting from the
        final layer.

        Note: Method is based of the BNN resample_parameters method.

        :param init_std: the standard deviation of the normal distribution from which we sample the
        weights. If None, the standard deviation of the original BNN is used.
        :param sparseness: the sparseness of the weights. If None, the sparseness of the original BNN
        is used.
        :param pre_activation_noise: the pre-activation noise of the BNN. If None,
        the sampled pre_activation noise is used. (Check the BNN's forward implementation for
        details)
        """
        bnn = self.relation_prior.model
        layers = bnn.linears[-n_layers:]

        init_std = init_std if init_std is not None else bnn.init_std
        sparseness = sparseness if sparseness is not None else bnn.sparseness

        for linear in layers:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in layers:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in layers[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

        if pre_activation_noise is None:
            # in every step of the forward, a rnd tensor is sampled, multiplied with the
            # pre_activation_noise and added to the x value. This changes the relation
            # TODO: check value for this!
            bnn.pre_activation_noise = np.random.uniform(0.0003, 0.0014)
        elif pre_activation_noise == 0:
            bnn.pre_activation_noise = 0

        if output_noise is None:
            bnn.output_noise = np.random.uniform(0.0004, 0.0013)
        elif output_noise == 0:
            bnn.output_noise = 0

    @property
    def configs(self):
        """Return the list of all possible configurations."""
        configs = {}
        config_list = []
        if self.sampling_seed is not None:
            config_list = self.sample(n=self.n_configs, seed=self.sampling_seed)
        else:
            config_list = self.sample(n=self.n_configs)
        for i, config in enumerate(config_list):
            configs[str(i)] = config
        return configs
    
    
    
if __name__ == "__main__":
    benchmark = SyntheticBenchmark(value_metric="value", cost_metric="fid_cost", seed=42)

    configs = benchmark.configs
    print(f"Number of Configs: {len(configs)}")
    print(f"Configs: {configs}")
    config = configs["0"]  # Get the first config
    print(f"Sampled Config: {config}")
    # result = benchmark.query(config, at=49)
    trajectory = benchmark.trajectory(
        config, frm=benchmark.start, to=benchmark.end)
    # np_config = np.array([config[f"X_{i}"]
    #                       for i in range(benchmark.dim_hyperparameters)])
    # curves = benchmark.relation_prior.curves_for_configs(np_config[np.newaxis, :], noise=False)
    # print(f"Curves: {curves(np.array([49/benchmark.max_fidelities]), 0)[0]}")
    # curves = benchmark.relation_prior.curves_for_configs(np_config[np.newaxis, :], noise=False)
    # print(f"Curves_2: {curves(np.array([49/benchmark.max_fidelities]), 0)[0]}")
    # print(f"Config: {config}")
    # print(f"Result: {result}")
    # print(f"Trajectory: {trajectory}")
    # print(f"Max Fidelity: {benchmark.end}")
    # print(f"Error: {benchmark.query(config, at=99).error}")
    related_benchmark = SyntheticBenchmark(value_metric="value", cost_metric="fid_cost", seed=42)
    
    # check if the model of the related benchmark is the same as the original benchmark
    if benchmark.relation_prior.model is not related_benchmark.relation_prior.model:
        print("Related benchmark has a different model than the original benchmark.")
    
    trajectory_related = related_benchmark.trajectory(
        config, frm=related_benchmark.start, to=related_benchmark.end)
    # plot the trajectory
    import matplotlib.pyplot as plt
    values = []
    values_related = []
    fidelities = np.arange(benchmark.start, benchmark.end + 1, 1)
    for f in trajectory:
        values.append(f.value.value)
    for f in trajectory_related:
        values_related.append(f.value.value)
    plt.plot(fidelities, values, marker='o')
    plt.plot(fidelities, values_related, marker='x')
    plt.xlabel("Fidelity (Epochs)")
    plt.ylabel("Value")
    plt.title("Synthetic Benchmark Trajectory")
    plt.legend(["Target Task", "Related Task"])
    plt.show()
