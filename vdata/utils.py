# coding: utf-8
# Created on 11/16/20 11:35 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import scipy.sparse as sp
from typing import Union, List, Any

from ._IO.errors import VValueError


# ====================================================
# code
def are_equal(a: Union[np.ndarray, sp.spmatrix], b: Union[np.ndarray, sp.spmatrix]) -> bool:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)

    elif isinstance(a, sp.spmatrix) and isinstance(b, sp.spmatrix):
        return (a != b).nnz == 0

    elif isinstance(a, sp.spmatrix):
        a = a.toarray()
        return np.array_equal(a, b)

    else:
        b = b.toarray()
        return np.array_equal(a, b)


def is_in(obj: np.ndarray, list_arrays: Union[np.ndarray, List[np.ndarray]]) -> bool:
    for arr in list_arrays:
        if are_equal(obj, arr):
            return True

    return False


def slice_to_range(s: slice, max_stop: Union[int, np.int_]) -> range:
    """
    Convert a slice to a range
    :param s: a slice to convert
    :param max_stop: if s.stop is None, the stop value for the range
    """
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else max_stop
    step = s.step if s.step is not None else 1

    return range(start, stop, step)


def isCollection(item: Any) -> bool:
    return True if hasattr(item, '__iter__') and not issubclass(type(item), str) else False


def to_list(value: Any) -> List[Any]:
    """
    Convert any object to a list.
    """
    if value is None:
        raise VValueError("Cannot cast 'None' to list.")

    if isCollection(value):
        return list(value)
    else:
        return [value]


def to_str_list(item: Any) -> List:
    """
    Converts a given object to a list of string (or list of list of string ...)
    :param item: an object to convert to list of string
    """
    new_tp_list: List[Union[str, List]] = []
    for v in to_list(item):
        if isCollection(v):
            new_tp_list.append(to_str_list(v))

        else:
            new_tp_list.append(str(v))

    return new_tp_list
