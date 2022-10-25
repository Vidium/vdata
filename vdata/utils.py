# coding: utf-8
# Created on 21/01/2021 11:21
# Author : matteo

# ====================================================
# imports
import builtins

import numpy as np
import pandas as pd
from collections.abc import MutableMapping, Collection

from typing import Union, Any, Sequence, MutableMapping as MutableMappingT, cast

from . import name_utils

# ====================================================
# code
_builtin_names = dir(builtins)
_builtin_names.remove('False')
_builtin_names.remove('True')
_builtin_names.remove('None')


# import sys
# from numbers import Number
# from collections import deque
# from collections.abc import Set, Mapping
# ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)
# def getsize(obj_0):
#     """Recursively iterate to sum size of object & members."""
#     _seen_ids = set()
#     def inner(obj):
#         obj_id = id(obj)
#         if obj_id in _seen_ids:
#             return 0
#         _seen_ids.add(obj_id)
#         size = sys.getsizeof(obj)
#         if isinstance(obj, ZERO_DEPTH_BASES):
#             pass # bypass remaining control flow and return
#         elif isinstance(obj, (tuple, list, Set, deque)):
#             size += sum(inner(i) for i in obj)
#         elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
#             size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
#         # Check for custom object instances - may subclass above too
#         if hasattr(obj, '__dict__'):
#             size += inner(vars(obj))
#         if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
#             size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
#         return size
#     return inner(obj_0)


# region misc -----------------------------------------------------------------
def get_value(v: Any) -> Union[str, int, float]:
    """
    If possible, get the int or float value of the passed object.
    :param v: an object for which to try to get the value.
    :return: the object's value (int or float) or the object itself.
    """
    v = str(v)

    if v in _builtin_names:
        return v

    try:
        v = eval(v)
        if isinstance(v, np.int_):
            return int(v)

        elif isinstance(v, np.float_):
            return float(v)

        else:
            return v

    except (NameError, SyntaxError):
        return v


def isCollection(obj: Any) -> bool:
    """
    Whether an object is a collection.
    :param obj: an object to test.
    :return: whether an object is a collection.
    """
    return isinstance(obj, Collection) and not isinstance(obj, (str, bytes, bytearray))


def are_equal(obj1: Any,
              obj2: Any) -> bool:
    from vdata.core.dataset_proxy import DatasetProxy

    if isinstance(obj1, (np.ndarray, DatasetProxy)):
        if isinstance(obj2, (np.ndarray, DatasetProxy)):
            return np.array_equal(obj1[:], obj2[:])

        return False

    return obj1 == obj2

# endregion


# region Representation --------------------------------------------------------------
def repr_array(arr: Union['name_utils.DType', Sequence, range, slice, 'ellipsis']) -> str:
    """
    Get a short string representation of an array.
    :param: an array to represent.
    :return: a short string representation of the array.
    """
    if isinstance(arr, slice) or arr is ... or not isCollection(arr):
        return str(arr)

    else:
        arr = cast(Sequence, arr)
        if isinstance(arr, range) or len(arr) <= 4:
            return f"{str(arr)} ({len(arr)} value{'' if len(arr) == 1 else 's'} long)"

        elif isinstance(arr, pd.Series):
            return f"[{arr[0]} {arr[1]} ... {arr.iloc[-2]} {arr.iloc[-1]}] ({len(arr)} values long)"

        else:
            return f"[{arr[0]} {arr[1]} ... {arr[-2]} {arr[-1]}] ({len(arr)} values long)"

# endregion


# region Type coercion ---------------------------------------------------------------
def deep_dict_convert(obj: MutableMappingT) -> Any:
    """
    'Deep' convert a mapping of any kind (and children mappings) into regular dictionaries.

    Args:
        obj: a mapping to convert.

    Returns:
        a converted dictionary.
    """
    if not isinstance(obj, MutableMapping):
        return obj

    return {k: deep_dict_convert(v) for k, v in obj.items()}

# endregion
