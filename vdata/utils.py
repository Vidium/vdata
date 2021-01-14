# coding: utf-8
# Created on 11/16/20 11:35 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
from typing import Union, List, Any, Collection

from ._IO.errors import VValueError


# ====================================================
# code
def is_in(obj: np.ndarray, list_arrays: Union[np.ndarray, List[np.ndarray]]) -> bool:
    for arr in list_arrays:
        if np.array_equal(obj, arr):
            return True

    return False


def slice_to_range(s: slice, max_stop: Union[int, np.int_]) -> range:
    """
    Converts a slice to a range
    :param s: a slice to convert
    :param max_stop: if s.stop is None, the stop value for the range
    """
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else max_stop
    step = s.step if s.step is not None else 1

    return range(start, stop, step)


def slice_to_list(s: slice, c: Collection[Any]) -> List[Any]:
    """
    Converts a slice to a list of elements within that slice.
    :param s: a slice to convert.
    :param c: a collection of elements to slice.
    """
    sliced_list = []
    found_start = False

    for e in c:
        if not found_start:
            if e == s.start:
                sliced_list.append(e)
                found_start = True

        else:
            if e == s.stop:
                break

            else:
                sliced_list.append(e)

    return sliced_list


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
