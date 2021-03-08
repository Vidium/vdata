# coding: utf-8
# Created on 21/01/2021 11:21
# Author : matteo

# ====================================================
# imports
import builtins
import numpy as np
import pandas as pd
from typing import Tuple, Union, Any, Sequence, cast

from . import NameUtils


# ====================================================
# code
_builtin_names = dir(builtins)
_builtin_names.remove('False')
_builtin_names.remove('True')
_builtin_names.remove('None')


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
        return eval(v)

    except (NameError, SyntaxError):
        return v


def isCollection(obj: Any) -> bool:
    """
    Whether an object is a collection.
    :param obj: an object to test.
    :return: whether an object is a collection.
    """
    return True if hasattr(obj, '__iter__') and not issubclass(type(obj), str) else False


# Representation ---------------------------------------------------------
def repr_array(arr: Union['NameUtils.DType', Sequence, range, slice, 'ellipsis']) -> str:
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


def repr_index(index: Union['NameUtils.PreSlicer', Tuple['NameUtils.PreSlicer'],
                            Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer'],
                            Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer', 'NameUtils.PreSlicer'],
                            Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
        -> str:
    """
    Get a short string representation of a sub-setting index.
    :param index: a sub-setting index to represent.
    :return: a short string representation of the sub-setting index.
    """
    if isinstance(index, tuple):
        repr_string = f"Index of {len(index)} element{'' if len(index) == 1 else 's'} : "

        for element in index:
            repr_string += f"\n  \u2022 {repr_array(element) if isCollection(element) else element}"

        return repr_string

    else:
        return f"Index of 1 element : \n" \
               f"  \u2022 {repr_array(index) if isCollection(index) else index}"
