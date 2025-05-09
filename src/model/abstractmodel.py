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

        if 'single_eval_pos' in kwargs:
            del kwargs['single_eval_pos']

        return self._forward(**kwargs)

    def _forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)