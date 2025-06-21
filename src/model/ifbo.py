from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd
import pfns4hpo

from metahyper import ConfigResult, instance_from_map

from neps.optimizers.multi_fidelity.dyhpo import MFEIBO
from neps.optimizers.bayesian_optimization.kernels.get_kernels import get_kernels
from neps.optimizers.multi_fidelity.mf_bo import FreezeThawModel, PFNSurrogate
from neps.utils.common import EvaluationData, SimpleCSVWriter
from neps.search_spaces.search_space import FloatParameter, IntegerParameter, SearchSpace
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.optimizers.bayesian_optimization.acquisition_functions.base_acquisition import \
    BaseAcquisition
from neps.optimizers.bayesian_optimization.acquisition_samplers import AcquisitionSamplerMapping
from neps.optimizers.bayesian_optimization.acquisition_samplers.base_acq_sampler import (
    AcquisitionSampler,
)
from neps.optimizers.bayesian_optimization.kernels.get_kernels import get_kernels
from neps.optimizers.multi_fidelity.mf_bo import FreezeThawModel, PFNSurrogate

# MFEIDeepModel, MFEIModel
from neps.optimizers.multi_fidelity.utils import MFObservedData

from neps.optimizers.bayesian_optimization.models.pfn import PFN_SURROGATE
import torch

from ifBO_icml2024.src.PFNs4HPO.pfns4hpo.model import BaseModel

from neps.search_spaces.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)

import logging

from src.model.mixture_ppd import PFNPPDMixture
from src.model.pfnimputation import PFNPriorImputation


class MyBaseModel(BaseModel):
    def __init__(self, path):
        torch.nn.Module.__init__(self)

        import importlib.util
        import sys

        # fixme: make this in reference to this file
        spec = importlib.util.spec_from_file_location("ifbo", f"{Path(path).parents[2]}/ifBO_main/ifbo/__init__.py")
        ifbo = importlib.util.module_from_spec(spec)
        sys.modules["ifbo"] = ifbo
        spec.loader.exec_module(ifbo)

        #        with torch.serialization.safe_globals([ifbo.transformer.TransformerModel]):

        self.model = torch.load(
                path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                weights_only=False
            )

        # import ifBO_icml2024
        #
        #
        #
        # with torch.serialization.safe_globals([
        #     ifBO_icml2024.src.PFNs4HPO.pfns4hpo.transformer.TransformerModel]):
        #     # (FUCK YOU GUYS with your absolute magic path name)
        #     self.model = torch.load(
        #         path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        #         weights_only=False
        #     )

        self.model.eval()

class MyPFN_MODEL(MyBaseModel, pfns4hpo.PFN_MODEL):
    def forward(self, x_train, y_train, x_test):
        if x_train.shape[0] == 0:
            x_test[:, 0] = 0
        elif x_train[:, 0].min() == 0:
            x_train[:, 0] += 1
            x_test[:, 0] += 1

            # reserve id=0 to curves that are not in x_train
            # set to 0 for all id in x_test[:, 0] that is not x_train[:, 0]
            x_test[:, 0] = torch.where(
                torch.isin(x_test[:, 0], x_train[:, 0]),
                x_test[:, 0],
                torch.zeros_like(x_test[:, 0]),
            )

        single_eval_pos = x_train.shape[0]
        batch_size = 2000
        n_batches = (x_test.shape[0] + batch_size - 1) // batch_size

        results = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, x_test.shape[0])
            x_batch = torch.cat([x_train, x_test[start:end]], dim=0).unsqueeze(1)
            y_batch = y_train.unsqueeze(1)
            result = self.model((x_batch, y_batch), single_eval_pos=single_eval_pos)
            results.append(result)

        final_result = torch.cat(results, dim=0)
        return final_result


class MyPFN_SURROGATE(PFN_SURROGATE):


    def __init__(
                self,
                pipeline_space: SearchSpace,
                logger=None,
                surrogate_model_fit_args: dict = None,
                model_name: str = None,
                minimize: bool = True,
                *args,
                **kwargs,  # pylint: disable=unused-argument
        ):


        self.minimize = minimize
        if model_name is None:
            self.model_name = kwargs['surrogate_model_args']['model_name']
        else:
            self.model_name = model_name
        self.__preprocess_search_space(pipeline_space)
        # set the categories array for the encoder
        self.categories_array = np.array(self.categories)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")

        # build the neural network
        self.nn = MyPFN_MODEL(self.model_name).to(self.device)
        self.logger = logger or logging.getLogger("neps")

    def __preprocess_search_space(self, pipeline_space: SearchSpace):
        self.categories = []
        self.categorical_hps = []

        parameter_count = 0
        for hp_name, hp in pipeline_space.items():
            # Collect all categories in a list for the encoder
            if isinstance(hp, CategoricalParameter):
                self.categorical_hps.append(hp_name)
                self.categories.extend(hp.choices)
                parameter_count += len(hp.choices)
            else:
                parameter_count += 1

        # add 1 for budget
        self.input_size = parameter_count + 1
        self.continuous_params_size = self.input_size - len(self.categories)

        self.min_fidelity = pipeline_space.fidelity.lower
        self.max_fidelity = pipeline_space.fidelity.upper


    def get_pi(self, x_test, inc, x_train=None, y_train=None):
        if isinstance(self.nn, PFNPriorImputation) or isinstance(self.nn, PFNPPDMixture):
            inc = inc.unsqueeze(1).to(self.device)
            return self.nn.get_pi(
                x_test=x_test,
                inc=((1 - inc) if self.minimize else inc),
                x_train=self.train_x if x_train is None else x_train,
                y_train=self.train_y if y_train is None else ((1 - y_train) if self.minimize else y_train),
                minimize=self.minimize,
            )

        else:
            return super(MyPFN_SURROGATE, self).get_pi(
                x_test=x_test,
                inc=inc,
                x_train=x_train,
                y_train=y_train,
            )

# overwrite the mapping to get rid of magic path
from neps.optimizers.bayesian_optimization.models import \
    SurrogateModelMapping

MySurrogateModelMapping = SurrogateModelMapping
MySurrogateModelMapping.update({'pfn': MyPFN_SURROGATE})



class MyFreezeThawModel(FreezeThawModel):

    def __init__(
            self,
            pipeline_space,
            surrogate_model: str = "deep_gp",
            surrogate_model_args: dict = None,
    ):
        self.observed_configs = None
        self.pipeline_space = pipeline_space
        self.surrogate_model_name = surrogate_model
        self.surrogate_model_args = (
            surrogate_model_args if surrogate_model_args is not None else {}
        )
        if self.surrogate_model_name in ["deep_gp", "pfn"]:
            self.surrogate_model_args.update({"pipeline_space": pipeline_space})
        elif self.surrogate_model_name == "dpl":
            self.surrogate_model_args.update(
                {"pipeline_space": self.pipeline_space,
                 "observed_data": self.observed_configs}
            )

        # instantiate the surrogate model
        self.surrogate_model = instance_from_map(
            MySurrogateModelMapping,
            self.surrogate_model_name,
            name="surrogate model",
            kwargs=self.surrogate_model_args,
        )

    def set_state(
        self,
        pipeline_space,
        surrogate_model_args,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.pipeline_space = pipeline_space
        self.surrogate_model_args = (
            surrogate_model_args if surrogate_model_args is not None else {}
        )
        if self.surrogate_model_name == "dpl":
            self.surrogate_model_args.update(
                {"pipeline_space": self.pipeline_space,
                 "observed_data": self.observed_configs}
            )
            self.surrogate_model = instance_from_map(
                SurrogateModelMapping,
                self.surrogate_model_name,
                name="surrogate model",
                kwargs=self.surrogate_model_args,
            )

        # only to handle tabular spaces
        if self.pipeline_space.has_tabular:
            if self.surrogate_model_name in ["deep_gp", "pfn"]:
                self.surrogate_model_args.update(
                    {"pipeline_space": self.pipeline_space.raw_tabular_space}
                )
            elif self.surrogate_model_name == "dpl":
                self.surrogate_model_args.update(
                    {"pipeline_space": self.pipeline_space,
                    "observed_data": self.observed_configs}
                )
            # instantiate the surrogate model, again, with the new pipeline space
            if self.surrogate_model_name in ['pfn', 'deep_gp', 'dpl']:
                self.surrogate_model = instance_from_map(
                    SurrogateModelMapping,
                    self.surrogate_model_name,
                    name="surrogate model",
                    kwargs=self.surrogate_model_args,
                )
            # else:
            #     # skip reinstantiation for custom models
            #     self.pipeline_space = self.surrogate_model_args['pipeline_space']
        elif self.surrogate_model_name == "dpl":
            self.surrogate_model_args.update(
                {"pipeline_space": self.pipeline_space,
                 "observed_data": self.observed_configs}
            )
            self.surrogate_model = instance_from_map(
                SurrogateModelMapping,
                self.surrogate_model_name,
                name="surrogate model",
                kwargs=self.surrogate_model_args,
            )


class MyPFNSurrogate(MyFreezeThawModel, PFNSurrogate):

    def _predict(self, test_x, test_lcs):
        # assert self.surrogate_model_name == "pfn" # killed the assert here
        test_x = self.preprocess_test_set(test_x)
        return self.surrogate_model.predict(self.train_x, self.train_y, test_x)

    def _fit(self, *args):  # pylint: disable=unused-argument
        # assert self.surrogate_model_name == "pfn" # killed the assert here
        self.preprocess_training_set()
        self.surrogate_model.fit(self.train_x, self.train_y)


class IFBO(MFEIBO):
    acquisition: str = "MFEI"

    def __init__(
            self,
            pipeline_space: SearchSpace,
            budget: int = None,
            step_size: int | float = 1,
            optimal_assignment: bool = False,  # pylint: disable=unused-argument
            use_priors: bool = False,
            sample_default_first: bool = False,
            sample_default_at_target: bool = False,
            loss_value_on_error: None | float = None,
            cost_value_on_error: None | float = None,
            patience: int = 100,
            ignore_errors: bool = False,
            logger=None,
            # arguments for model
            surrogate_model: str | Any = "deep_gp",
            surrogate_model_args: dict = None,
            domain_se_kernel: str = None,
            graph_kernels: list = None,
            hp_kernels: list = None,
            acquisition: str | BaseAcquisition = acquisition,
            acquisition_args: dict = None,
            acquisition_sampler: str | AcquisitionSampler = "freeze-thaw",
            acquisition_sampler_args: dict = None,
            model_policy: Any = FreezeThawModel,
            initial_design_fraction: float = 0.75,
            initial_design_size: int = 10,
            initial_design_budget: int = None,
    ):
        """Initialise

               Args:
                   pipeline_space: Space in which to search
                   budget: Maximum budget
                   use_priors: Allows random samples to be generated from a default
                       Samples generated from a Gaussian centered around the default value
                   sampling_policy: The type of sampling procedure to use
                   promotion_policy: The type of promotion procedure to use
                   loss_value_on_error: Setting this and cost_value_on_error to any float will
                       supress any error during bayesian optimization and will use given loss
                       value instead. default: None
                   cost_value_on_error: Setting this and loss_value_on_error to any float will
                       supress any error during bayesian optimization and will use given cost
                       value instead. default: None
                   logger: logger object, or None to use the neps logger
                   sample_default_first: Whether to sample the default configuration first
               """
        BaseOptimizer.__init__(self,
            pipeline_space=pipeline_space,
            budget=budget,
            patience=patience,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            ignore_errors=ignore_errors,
            logger=logger,
        )
        self.raw_tabular_space = None  # placeholder, can be populated using pre_load_hook
        self._budget_list: list[int | float] = []
        self.step_size: int | float = step_size
        self.min_budget = self.pipeline_space.fidelity.lower
        # TODO: generalize this to work with real data (not benchmarks)
        self.max_budget = self.pipeline_space.fidelity.upper

        self._initial_design_fraction = initial_design_fraction
        self._initial_design_size, self._initial_design_budget = self._set_initial_design(
            initial_design_size, initial_design_budget, self._initial_design_fraction
        )
        # TODO: Write use cases for these parameters
        self._model_update_failed = False
        self.sample_default_first = sample_default_first
        self.sample_default_at_target = sample_default_at_target

        if isinstance(surrogate_model, str):
            self.surrogate_model_name = surrogate_model
        else:
            self.surrogate_model_name = surrogate_model.__name__

        self.use_priors = use_priors
        self.total_fevals: int = 0

        self.observed_configs = MFObservedData(
            columns=["config", "perf", "learning_curves"],
            index_names=["config_id", "budget_id"],
        )

        # Preparing model
        self.graph_kernels, self.hp_kernels = get_kernels(
            pipeline_space=pipeline_space,
            domain_se_kernel=domain_se_kernel,
            graph_kernels=graph_kernels,
            hp_kernels=hp_kernels,
            optimal_assignment=optimal_assignment,
        )
        self.surrogate_model_args = (
            {} if surrogate_model_args is None else surrogate_model_args
        )
        self._prep_model_args(self.hp_kernels, self.graph_kernels, pipeline_space)

        # TODO: Better solution than branching based on the surrogate name is needed
        if isinstance(surrogate_model, str):
            if surrogate_model in ["deep_gp", "gp", "dpl"]:
                model_policy = FreezeThawModel
            elif surrogate_model == "pfn":
                model_policy = MyPFNSurrogate
            else:
                raise ValueError("Invalid model option selected!")

            # The surrogate model is initalized here
            self.model_policy = model_policy(
                pipeline_space=pipeline_space,
                surrogate_model=surrogate_model,
                surrogate_model_args=self.surrogate_model_args,
            )
        else:
            self.model_policy = surrogate_model

        self.acquisition_args = {} if acquisition_args is None else acquisition_args
        self.acquisition_args.update(
            {
                "pipeline_space": self.pipeline_space,
                "surrogate_model_name": self.surrogate_model_name,
            }
        )
        self.acquisition = instance_from_map(
            AcquisitionMapping,
            acquisition,
            name="acquisition function",
            kwargs=self.acquisition_args,
        )
        self.acquisition_sampler_args = (
            {} if acquisition_sampler_args is None else acquisition_sampler_args
        )
        self.acquisition_sampler_args.update(
            {"patience": self.patience, "pipeline_space": self.pipeline_space}
        )
        self.acquisition_sampler = instance_from_map(
            AcquisitionSamplerMapping,
            acquisition_sampler,
            name="acquisition sampler function",
            kwargs=self.acquisition_sampler_args,
        )
        self.count = 0

        self.evaluation_data = EvaluationData()

    def get_config_and_ids(  # pylint: disable=no-self-use
        self,
    ) -> tuple[SearchSpace, str, str | None]:

        # FIXME: overwrite this for the acquisition imputation idea
        return super().get_config_and_ids()