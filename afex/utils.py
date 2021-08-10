import torch
import numpy as np
from typing import Union, Tuple, Callable, Iterator, List, Dict, TypeVar
from collections import defaultdict


def sort_tensor_values(input_xs: torch.Tensor):
    values, _indices = torch.sort(input_xs)
    return values


IntOrTuple = Union[int, Tuple[int, ...]]


def deep_divide(divisor: int) -> Callable[[IntOrTuple], IntOrTuple]:
    """Make a function which will divide its argument by `divisor`.
    If it is a tuple, then divide each its element.

    :param divisor: Divisor (denominator).
    :return: Division function.
    """

    def _divide(item):
        if isinstance(item, tuple):
            return tuple(numerator // divisor for numerator in item)
        return item // divisor

    return _divide


K = TypeVar('K')
E = TypeVar('E')
V = TypeVar('V')


def collapse(kv_generator: Iterator[Tuple[K, V]],
             key_reducer: Callable[[K], E]) -> Dict[E, List[V]]:
    """Collapse (k, v) generator by its key, collecting all values
       corresponding to each `key_reducer(k)` in lists.

    :param kv_generator: Key-value generator, e.g. `dict.items()`.
    :param key_reducer: Function which maps keys to their equivalence classes representatives.
    """
    collapsed = defaultdict(lambda: [])
    for k, v in kv_generator:
        collapsed[key_reducer(k)].append(v)
    return collapsed


def min_max_normalize(values: Union[list, np.ndarray]) -> np.ndarray:
    result = np.array(values)
    result -= result.min()
    result_max = result.max()
    if abs(result_max) > 1e-38:
        result /= result.max()
    return result

