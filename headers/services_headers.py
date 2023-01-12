from typing import Iterable, Dict, Tuple, Union, Any


class CoresetServiceLG:
    """
    Service class for working with a logistic regression coreset.
    """
    ...

    def build(
            self,
            X: Iterable,
            y: Iterable = None,
            indices: Iterable = None,
    ) -> 'CoresetServiceLG':
        """
        Create a coreset from a transformed dataset(s).

        Parameters
        ----------
        X: array-like.
        y: array-like, optional.
        indices: array-like, optional, default sequence.
            Allow setting own indices.

        Returns
        -------
        self
        """

    def get_important_samples(
            self,
            size: int = None,
            class_size: Dict[Any, Union[int, str]] = None,
            *,
            ignore_indices: Iterable = None
    ) -> Tuple[Iterable[int], Iterable[float]]:
        """
        Returns indices of most important samples order by importance.
        Useful for identifying and fixing incorrectly labeled data as well as
        identifying other anomalies in your dataset such as data imbalance and under-represented.

        Parameters
        ----------
        size: int, optional
            Number of samples to return.
            When class_size is provided, remaining samples are taken from classes not appearing in class_size dictionary.

        class_size: dict {class: int or "all" or "any"}, optional.
            Controls the number of samples to choose for each class.
            int: return at most size.
            "all": return all samples.
            "any": return any number of instances up to size.

        ignore_indices: array-like, optional.
            An array of indices to ignore when selecting important samples.

        Returns
        -------
        tuple:
            indices: array-like[int].
                important samples indices.
            importance: array-like[float].
                The important value. High value is more important.
                The importance values. High value is more important.

        Examples
        -------
        Input:
            size=100,
            class_size={"class A": 10, "class B": 50, "class C": "all"}
        Output:
            10 of "class A",
            50 of "class B",
            12 of "class C" (all),
            28 of "class D/E"
        """
