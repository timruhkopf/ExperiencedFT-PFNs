import torch

from model.weights.abstract_weights import AbstractWeights

def override_call_decorator(func):
    def wrapper(x_train, x_test, *args, **kwargs):
        n_x_test = x_test.shape[0]

        # we must consider, that we ask for the next fidelity of seen configurations
        train = x_train.clone()
        min_fidelity = train[:, :, 1].min()
        train[:, :, 1] += min_fidelity

        # we need to also consider, that in x_test we ask unseen configurations
        test = x_test.clone()
        test[:, :, 1] = min_fidelity

        # METHOD OVERRIDE!
        result_logits = func(
            x_train=x_train,
            x_test=torch.cat([x_test, train, test], dim=0),
            *args, **kwargs
        )
        test_logits = result_logits[:n_x_test, :, :]
        lookahead_logits = result_logits[n_x_test:, :, :]

        # gain access to the parent model
        self = getattr(func, '__self__', None)
        self.parent_model.interim_results.update({
            f'last-{func.__name__}-lookahead': lookahead_logits.cpu(),
            f'x_lookahead': torch.cat([train[:, 0:1, :], test[:, 0:1, :]], dim=0).cpu()

        })

        return test_logits

    return wrapper

class BeliefChange(AbstractWeights):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self, related_context, device, logger, model, parent_model):
        super().__post_init__(
            related_context=related_context,
            device=device,
            logger=logger,
            model=model,
            parent_model=parent_model
        )

        # for efficiency reasons, we decorate the model call to compute the one step lookahead
        # logits, so we can have a check what we believed about y for the next step.
        self.parent_model.strategy.get_target_model = override_call_decorator(
            self.parent_model.strategy.get_target_model
        )
        self.parent_model.strategy.get_imputation_augmented_prior = override_call_decorator(
            self.parent_model.strategy.get_imputation_augmented_prior
        )
        if hasattr(self.parent_model.strategy, 'get_error_model'):
            self.parent_model.strategy.get_error_model = override_call_decorator(
                self.parent_model.strategy.get_error_model
            )

        if hasattr(self.parent_model.strategy, 'get_prior_augmented_target_model'):
            self.parent_model.strategy.get_prior_augmented_target_model = override_call_decorator(
                self.parent_model.strategy.get_prior_augmented_target_model
            )

        self.parent_model.interim_results.update({
            'surprise_logits': [],
            'surprise_x': [],
            'surprises_nll': [],
            'last_step_logits': None,
            'past_x_test': None,

        })

    @staticmethod
    def find_extra_row_index(A, B):
        """
        Find the row not present in B, assuming A has exactly one extra row,
        whose location is unknown.

        :example:
            i=3
            A=torch.randn(10,3)
            B = torch.cat([A.clone()[:i, :], torch.ones(1, 3), A.clone()[i:, :]])
            assert i == find_extra_row_index(B,A)
        :param A:
        :param B:
        :return:
        """
        a = set(tuple(v) for v in A.tolist())
        b = set(tuple(v) for v in B.tolist())
        new_idx = torch.tensor(list(b.difference(a))).to(A.device)

        return torch.where((B == new_idx).all(dim=-1).flatten())[0].item()

    def __call__(self, x_train, x_test, y_train, inc, *args, **kwargs):

        # append last x_test to the query and check what the difference in belief of
        # PI utility is when comparing both logits under last step's incumbent

        # 1. get last step's test logits and x_test
        last_step_logits = self.parent_model.interim_results.get('last_step_logits', None)
        past_x_test = self.parent_model.interim_results.get('past_x_test', None)

        # 2. under the current target state collect the logits for the last step's x_test
        # here we need to evaluate the model on the last step's x_test

        # the new belief is the actual change in expected utility for the x_test (i.e. reducing the
        # future horizon by 1 step)
        new_belief = self.parent_model.strategy.get_target_model(
            x_train=x_train,
            x_test=past_x_test,
            y_train=y_train,
            *args, **kwargs
        )


        # 3. compute the PI utility for both logits under the last step's incumbent
        last_inc = self.parent_model.interim_results.get('last_inc', None)



        self.parent_model.interim_results.update({
            'last_inc': inc,
        })
        return