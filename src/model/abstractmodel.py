import torch


class AbstractModel:
    def pre_train(self, tasks):
        pass

    def train(self, *args, **kwargs):
        """
        returns the prefix (x,y) for the new task; necessary for the test_on_task to comply
        with the api in case of distillation
        """
        return None

    def forward(self, *args, **kwargs):
        """
        Interface for the TransformerModel class from ifbo.
        """
        if isinstance(args, tuple) and len(args) == 1:
            # this is the unfortunate TransformerModel compatability
            args = args[0]
            x, y = args
            single_eval_pos = kwargs['single_eval_pos']
            kwargs = {
                'context_x': x[:single_eval_pos],
                'context_y': y,
                'query_x': x[single_eval_pos:]
            }
        elif set(kwargs.keys()) == {'x_train', 'y_train', 'x_test'}:
            # during ifbo deployment
            # curve id encoder will struggle with index positions longer then the
            # sequence length --> so we just replace them here. They don't have any meaning anyways!
            mask = kwargs['x_test'][:, 0] >= 1000
            kwargs['x_test'][mask, 0] = torch.tensor(999.)

            kwargs = {
                'context_x': kwargs['x_train'].unsqueeze(1),
                'context_y': kwargs['y_train'].unsqueeze(1),
            'query_x': kwargs['x_test'].unsqueeze(1)
            }


        if 'single_eval_pos' in kwargs:
            del kwargs['single_eval_pos']

        return self._forward(**kwargs)

    def _forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)