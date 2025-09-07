from src.model.strategies.abstract_strategy import AbstractStrategy


class AbstractInitialDesign(AbstractStrategy):
    def __init__(self, size,):
        self.size = size
