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
        "value": Metric(minimize=False, bounds=(0.0, 1.0))
    }
    default_value_metric: ClassVar[str] = "value"
    default_value_metric_test: ClassVar[None] = None
    default_cost_metric: ClassVar[None] = None

    value: Metric.Value


C = TypeVar("C", bound=Config)
R = TypeVar("R", bound=Result)


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

    ):

        _max_fidelity = 100  # Maximum number of epochs

        # sample the dimensionality (i.e. hp space dim)
        self.dim_hyperparameters = 6

        # set up the BNN mapping from the hyperparameter space to the learning curves
        n_curve_param = 23  # Number of parameters for the learning curve basis and their weights
        self.relation_prior = DatasetPrior(
            self.dim_hyperparameters, n_curve_param)
        self.relation_prior.new_dataset()

        name = (
            f"SyntheticBenchmark_{self.dim_hyperparameters}D"
        )

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
            fidelity_range=(0, _max_fidelity, 1),
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
            value_metric=value_metric,
        )

    @override
    def _objective_function(
        self, config: Mapping[str, Any], *, at: int,
    ) -> dict[str, float]:
        # convert config to np array
        config = np.array([config[f"X_{i}"]
                            for i in range(self.dim_hyperparameters)])
        
        curves = self.relation_prior.curves_for_configs(config[np.newaxis,:])

        return {"value": curves(np.linspace(0, 1, 100), 0)[at]} # what should be the first argument? 


if __name__ == "__main__":
    benchmark = SyntheticBenchmark(value_metric="value")
    
    config = benchmark.sample()
    print(f"Sampled Config: {config}")
    result = benchmark.query(config, at=50)  # Query at 50 epochs
    trajectory = benchmark.trajectory(config, frm=5, to=50)
    print(f"Config: {config}")
    print(f"Result: {result}")
    print(f"Trajectory: {trajectory}")
    print(f"Max Fidelity: {benchmark.end}")