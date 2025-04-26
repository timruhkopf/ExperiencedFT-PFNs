from typing import Callable, Union, List

import numpy as np
import torch

from ifbo.priors.ftpfn_prior import DatasetPrior
from ifbo.utils import detokenize, Curve

from ifbo import Batch


class MetaTaskPrior:
    def __init__(self, dim_hyperparameters, n_fidelities=None, seq_len=1000,
                 device='cpu'):

        # sample the dimensionality (i.e. hp space dim)
        self.dim_hyperparameters = np.random.randint(1, dim_hyperparameters - 1)

        self.n_fidelities = n_fidelities if n_fidelities is not None else \
            int(np.round(10 ** np.random.uniform(0, 3)))
        self.seq_len = seq_len

        # set up the BNN mapping from the hyperparameter space to the learning curves
        n_curve_param = 23  # Number of parameters for the learning curve basis and their weights
        self.relation_prior = DatasetPrior(self.dim_hyperparameters, n_curve_param)
        self.relation_prior.new_dataset()

        self.device = device

    def __hash__(self):
        self.relation_prior.model.__hash__()

    def sample_dirichlet(self, alpha: float, eps=10 ** -9, single_eval_pos: int = 500) -> \
            (np.ndarray, np.ndarray, np.ndarray):
        """
        Sample a Dirichlet distribution to sample a budget allocation along the curves for both 
        context and query.
        
        :param alpha: the concentration parameter of the Dirichlet distribution.
         smaller α → sparser allocations (focus on few configs): Exploitation-focused (low α →
            few configs get most tokens). Exploration-focused with larger α → uniform distribution.
        :param eps: some small value to avoid division by zero
        :param single_eval_pos: given the sequence length of tokens, this is the position where we
         split context and query. The first part of the sequence is the context, the second part is the query.
        :return: Tuple:
            - cutoff_per_curve: the number of observations for each curve
            - epochs_per_curve: the number of queries for each curve
            - ordering: the ordering of the tokens
        """
        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        # Gamma sampling converts to Dirichlet distribution via normalization
        # Smaller α → sparser allocations (focus on few configs):  Exploitation-focused (low α →
        # few configs get most tokens)
        # Larger α → uniform distribution (equal attention): Exploration-friendly
        if alpha is None:
            alpha = 10 ** np.random.uniform(-4, -1)

        weights = np.random.gamma(alpha, alpha, self.seq_len) + eps
        p = weights / np.sum(weights)

        ids = np.arange(self.seq_len)  # identify, which token belongs to which curve
        all_levels = np.repeat(ids,
                               self.n_fidelities)  # since each curve has self.n_fidelities (fidelities) we
        # could observe, this is just a flat array for all curves after one another
        all_p = np.repeat(p, self.n_fidelities) / self.n_fidelities
        # The ordering vector is basically the mapping to the hp idx for each token
        # provided, that the token ordering is random and we are having a cutoff point where the
        # query starts, the subsequent for loop will do a "cumsum" over the occurences of the idx
        # to determine the length of that curve and where we are querying (past the query cutoff)
        ordering = np.random.choice(all_levels, p=all_p, size=self.seq_len, replace=False)

        cutoff_per_curve = np.zeros((self.seq_len,), dtype=int)
        epochs_per_curve = np.zeros((self.seq_len,), dtype=int)
        for i in range(self.seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        return cutoff_per_curve, epochs_per_curve, ordering

    def sample_hyperparameters(self, ):
        return np.random.uniform(size=(self.seq_len, self.dim_hyperparameters))

    def get_curves(self, hps) -> Callable:
        """Given a sampled instantiation of a relation, we can collect the curves for
         specific hyperparameter configurations."""
        return self.relation_prior.curves_for_configs(hps)

    def _interpret_dirichlet_sample(
            self,
            ordering,
            epochs_per_curve,
            cutoff_per_curve,
            hps,
            single_eval_pos
    ):
        """
        Interpret the Dirichlet sample and create the batch data element.
        :param ordering: index vector, mapping tokens to hp configs (of length self.seq_len),
        containing both context and query tokens.
        :param epochs_per_curve: the number of queries for each curve
        :param cutoff_per_curve: The number of observations for each curve (i.e. for each index/hp)
        :param hps: the hyperparameters for each index
        :param single_eval_pos: the position where the query starts
        :return:
        """

        epoch = torch.zeros(self.seq_len)
        id_curve = torch.zeros(self.seq_len)
        curve_val = torch.zeros(self.seq_len)
        config = torch.zeros(self.seq_len, self.dim_hyperparameters)

        curves = self.relation_prior.curves_for_configs(hps)
        curve_xs = []  # for each hp, which x's need evaluation
        curve_ys = []  # for each hp, the y's that are evaluated (incl. context & query
        for cid in range(self.seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                # determine x (observations + query)
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[: cutoff_per_curve[cid]] = (
                            np.arange(1, cutoff_per_curve[cid] + 1) / self.n_fidelities
                    )
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    # Query points are uniformly selected along the unobserved curve
                    # -- they are non-consecutively queried anywhere along that curve.
                    x_[cutoff_per_curve[cid]:] = (
                            np.random.choice(
                                np.arange(cutoff_per_curve[cid] + 1, self.n_fidelities + 1),
                                size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                                replace=False,
                            )
                            / self.n_fidelities
                    )
                curve_xs.append(x_)
                # determine y's
                y_ = curves(x_, cid)
                curve_ys.append(y_)
            else:
                curve_xs.append(None)
                curve_ys.append(None)

        # construct the batch data element

        curve_counters = torch.zeros(self.seq_len).type(torch.int64)
        for i in range(self.seq_len):
            cid = ordering[i]
            if i < single_eval_pos or curve_counters[cid] > 0:
                id_curve[i] = cid + 1  # reserve ID 0 for queries
            else:
                id_curve[i] = 0  # queries for unseen curves always have ID 0
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            config[i] = torch.from_numpy(hps[cid])
            curve_val[i] = curve_ys[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        x = torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1)
        y = curve_val

        return x, y

    def sample_related_task(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def sample_batch(self, n_tasks, alphas: Union[List[float], float] = None,
                     single_eval_pos: Union[List[
                         int], int] = None) -> Batch:
        """
        Sample a batch of related tasks.
        :param n_tasks: the number of related tasks to sample
        :param alphas: List or float of the concentration parameter of the Dirichlet distribution
        for each task.
         if it is an int, it will be used for all tasks.
        :param single_eval_pos: the context size for each task. If it is an int, it will be used for all tasks.

        :return:
        Batch
           Named tuple containing:
           - x: Input tensor of shape (seq_len, batch_size, dim_hyperparameters + 2 -- i.e.
           prepended additional dimensions: id_curve and epoch)
               Contains concatenated [id_curve, epoch, config_parameters]
           - y: Target values tensor of shape (seq_len, batch_size)
           - target_y: Copy of y for compatibility
           - single_eval_pos: List of context sizes for each task
        """

        if alphas is None:
            alphas = [10 ** np.random.uniform(-4, -1) for _ in range(n_tasks)]
        if isinstance(alphas, float):
            alphas = [alphas] * n_tasks
        assert len(alphas) == n_tasks, "alphas must be a list of the same length as n_tasks"

        if single_eval_pos is None:
            single_eval_pos = int(np.round(self.seq_len / 2))
        if isinstance(single_eval_pos, int):
            single_eval_pos = [single_eval_pos] * n_tasks
        assert len(
            single_eval_pos) == n_tasks, "single_eval_pos must be a list of the same length as n_tasks"

        self.relation_prior.new_dataset()

        X = []; Y = []
        for task, alpha, context_size in zip(range(n_tasks), alphas, single_eval_pos):
            self.sample_related_task()
            x, y = self.sample_from_task(
                alpha=alpha,
                context_size=context_size
            )

            X.append(x)
            Y.append(y)

        X = torch.stack(X, dim=1).to(self.device).float()
        Y = torch.stack(Y, dim=1).to(self.device).float()

        return Batch(x=X, y=Y, target_y=Y.clone(), single_eval_pos=single_eval_pos)

    def sample_from_task(self, alpha, context_size):
        # sample exactly seq_len hps, but some may become inactive
        hps = self.sample_hyperparameters()
        cutoff_per_curve, epochs_per_curve, ordering = self.sample_dirichlet(
            alpha=alpha,
            single_eval_pos=context_size,
        )
        x, y = self._interpret_dirichlet_sample(
            ordering=ordering,
            epochs_per_curve=epochs_per_curve,
            cutoff_per_curve=cutoff_per_curve,
            hps=hps,
            single_eval_pos=context_size
        )
        return x, y


class MetaTaskPriorSameProblem(MetaTaskPrior):
    def sample_related_task(self):
        """
        Sample a related task. In this case, we stick with the same problem, but sample both a new
        set of hyperparameters and a new budget allocation. Notice, that even then the task is not
        exactly the same, since the bnn will add layerwise noise.
        """
        pass


class MetaTaskPriorResampleLayers(MetaTaskPrior):
    def __init__(
            self,
            n_layers,
            reset_kwargs={
                "init_std": None,
                "sparseness": None,
                "pre_activation_noise": 0,
                "output_noise": 0
            },
            *args, **kwargs
    ):
        """

        :param n_layers: determines the number of layers (starting from the output layer) that we are
        resampling from the bnn, to establish a new task.
        :param reset_kwargs: a dictionary containing the parameters for the resampling:
            - init_std: the standard deviation of the normal distribution from which we sample the
            weights. If None, the standard deviation of the original BNN is used.
            - sparseness: the sparseness of the weights. If None, the sparseness of the original BNN
            is used.
            - pre_activation_noise: the pre-activation noise of the BNN. If None,
            the sampled pre_activation noise is used. (Check the BNN's forward implementation for
            details)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        self.n_layers = n_layers
        self.reset_kwargs = reset_kwargs

    def sample_related_task(self):
        """
        Convenience wrapper to make the api more consistent.
        Sample a related task. In this case, we stick with the same problem, but sample both a new
        set of hyperparameters and a new budget allocation. Notice, that even then the task is not
        exactly the same, since the bnn will add layerwise noise.
        """
        self._sample_related_task(**self.reset_kwargs)

    def _sample_related_task(
            self,
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
        layers = bnn.linears[-self.n_layers:]

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


def detokenize_batch(batch: Batch) -> List[List[Curve]]:
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
    meta_task_prior = MetaTaskPriorSameProblem(
        dim_hyperparameters=3,

        n_fidelities=100,
        seq_len=1000,
        device='cpu'
    )

    # Sample a batch of related tasks
    batch = meta_task_prior.sample_batch(n_tasks=5, alphas=0.1, single_eval_pos=500)

    # detokenize the batch
    # fixme: detokenize function splits based on an int context_size for all tasks equally

    assert all(batch.single_eval_pos[0] == i for i in batch.single_eval_pos), \
        "detokenize function does not support different context sizes for each task of a batch"

    try:
        # FIXME: detokenize works only with batch sizes of 1!
        context_list, query_list = detokenize(batch, context_size=500, device='cpu')
    except RuntimeError as e:
        print("detokenize function does not support batch sizes > 1, splitting the batch into "
              "single batches")
        print(e)

        contexts=detokenize_batch(batch)

    print()
