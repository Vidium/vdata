# coding: utf-8
# Created on 21/01/2021 11:21
# Author : matteo

# ====================================================
# imports
import builtins

import anndata.compat
import numpy as np
import pandas as pd
from collections import abc

from typing import Union, Any, Sequence, MutableMapping, cast

from . import name_utils


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
    return True if hasattr(obj, '__iter__') and not issubclass(type(obj), str) else False


# Representation --------------------------------------------------------------
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


# Type coercion ---------------------------------------------------------------
def deep_dict_convert(obj: MutableMapping) -> Any:
    """
    'Deep' convert a mapping of any kind (and children mappings) into regular dictionaries.

    Args:
        obj: a mapping to convert.

    Returns:
        a converted dictionary.
    """
    if not isinstance(obj, abc.MutableMapping):
        return obj

    return {k: deep_dict_convert(v) for k, v in obj.items()}
