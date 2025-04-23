import torch


class AbstractModel:
    def pre_train(self, tasks):
        pass

    def train(self, tasks):
        """
        returns the prefix (x,y) for the new task; necessary for the test_on_task to comply
        with the api in case of distillation
        """
        return torch.tensor([]), torch.tensor([])

    def forward(self):
        pass