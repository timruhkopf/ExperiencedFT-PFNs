from neps.optimizers.multi_fidelity.dyhpo import MFEIBO


class IFBO(MFEIBO):
    pass


class IFBO_mixture(MFEIBO):
    def __init__(
            self,
            pipeline_space,
            max_cost_total,
            acquisition,
            acquisition_args,
            acquisition_sampler,
            acquisition_sampler_args,
            surrogate_model,
            surrogate_model_args,
            model_name,
            checkpointing,
            root_directory,
            initial_design_fraction,
            initial_design_size,
            initial_design_budget,
            loss_value_on_error,
            cost_value_on_error,
            ignore_errors):
        pass

        # TODO instantiate the PFN surrogate model
        # TODO in some method pass it optionally the benchmerk data
