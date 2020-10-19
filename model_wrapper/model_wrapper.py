import itertools
from typing import List, Optional, Any
from sklearn.model_selection import train_test_split


class _KVPair:
    """
    _KVPair is a private helper class for ModelWrapper, the class allows for easy access to values without needing the
    key. The easy access is why a dictionary could not be used

    :param key: The "key" between the two objects associated with the object
    :param value: The "value" between the two objects associated with the object, used to do all of comparing
    """

    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value

    def __ge__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return f'{self.value} : {self.value}'


class ModelWrapper:
    """
    ModelWrapper will wrap around AI/ML Models from different packages and will allows th user to run various different
    tests.

    :param model_class: The model_class that will be wrapped around.
    :param dataset: The dataset the model will be trained with and tested.
    :param train_percent: the percentage of the dataset that will be used to train the model
    :param test_percent: the percentage of the dataset that will be used to test and score the model
    """

    def __int__(self, model_class: type, dataset, train_percent: float = .80, test_percent: float = .20,
                random_state: int = 42):
        self.model_class = model_class
        self.dataset = dataset
        self._best_pair = None

        # random_sate allows for same results to be achieved everytime
        self._train_x, self._test_x, self._train_y, self._test_y = train_test_split(dataset, test_size=test_percent,
                                                                                    random_state=random_state)

    def score(self) -> Optional[Any]:
        """
        This function will score the currently wrapped model.

        :return: A single mapping of the parameters in a tuple mapped to the highest score the model has reached
        """
        return self._best_pair

    def explore(self, result_size: int = float('inf'), **kwargs) -> List[_KVPair]:
        """
        Function which completely trains the given model with the given dataset. Composed of various parameters for
        testing and fine-tuning purposes.

        :param c_list: A list of c values to be tested with
        :param gamma_list: A list of gamma values to be tested with
        :param result_size: the max size of the returned list
        :return: A sorted list of size result_size with mappings of the params used to obtain the mapped score sorted
            by the mapped score
        """

        explore_results = self._explore_helper(**kwargs)
        count = -1
        results = []
        for y in sorted(explore_results):
            count += 1
            if count < result_size:
                results.append(y)
            else:
                return results

        return results

    def _explore_helper(self, **kwargs) -> List[_KVPair]:
        """
        Function which computes values after completing training iterations based on the given model with the given
        dataset

        :param kwargs: A dictionary where the key is the name of the parameter and the value is a list of all values to
        be explored
        """

        unsorted_results = []

        list_of_possible_args = []
        for param_label, param_values in kwargs.items():
            list_of_possible_args.append([{param_label: x} for x in param_values])

        for param_set in itertools.product(*list_of_possible_args):
            params = {}
            for pair in param_set:
                params.update(pair)

            model = self.model_class(**params)
            model.fit(self._train_x, self._test_y)
            new_pair = _KVPair(params, self.model.score(self._test_x, self._test_y))

            if not self._best_pair or new_pair > self._best_pair:
                self._best_pair = new_pair

            unsorted_results.append(new_pair)

        return unsorted_results
