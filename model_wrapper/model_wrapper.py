from typing import List, Dict


class ModelWrapper:
    """
    This is a class for mathematical operations on complex numbers.

    Attributes:
        model : The model that will be wrapped around.
        dataset : The dataset the model will be trained with and tested.
        train_percent (float): the percentage of the dataset

    """

    def __int__(self, model, dataset, train_percent: float = .80, test_percent: float = .20):
        self.model = model
        self.dataset = dataset
        self.train_percent = train_percent
        self.test_percent = test_percent

    # TODO add docstring
    def score(self) -> Dict[tuple, float]:
        """"""
        pass

    # TODO add docstring
    def explore(self, c: List[float], gamma: List[float], result_size: int = 10) -> Dict[tuple, float]:
        """"""
        pass

    def _explore_helper(self, c: List[float], gamma: List[float], result_size: int = 10) -> Dict[tuple, float]:
        """"""
        pass





"""

AI/ML Algorithm Analysis
Chase Crossley, Rigoberto Gonzalez, Jacob Nona and, Timothy Roe

"""
