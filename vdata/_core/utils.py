# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Collection, Optional

from ..NameUtils import PreSlicer, ArrayLike_2D
from .._IO.errors import VTypeError, ShapeError


# ====================================================
# code
def format_index(index: Union[PreSlicer, Tuple[PreSlicer, PreSlicer], Tuple[PreSlicer, PreSlicer, PreSlicer]]) \
        -> Tuple[PreSlicer, PreSlicer, PreSlicer]:
    """
    Simple function for formatting a sub-setting index : it makes sure the index has the format (..., ..., ...).
    :param index: an index to format.
    :return: a formatted index.
    """
    if not isinstance(index, tuple):
        index = (index, ..., ...)
    elif len(index) == 2:
        index = (index[0], index[1], ...)

    # get slicers
    time_points_slicer = index[0]
    obs_slicer = index[1]
    var_slicer = index[2]

    return time_points_slicer, obs_slicer, var_slicer


def repr_array(arr: Union[List, np.ndarray]) -> str:
    """
    Get a short string representation of an array.
    :param: an array to represent.
    :return: a short string representation of the array.
    """
    if len(arr) > 6:
        return f"[{arr[0]} {arr[1]} {arr[2]} ... {arr[-3]} {arr[-2]} {arr[-1]}]"

    else:
        return str(arr)


# TODO : update
def reshape_to_3D(arr: ArrayLike_2D, time_points: Optional[Collection], time_list) -> np.ndarray:
    """
    Reshape a 2D array-like object into a 3D array-like. Pandas DataFrames are first converted into numpy arrays.
    A time points collection (of the same length as the number of rows in the 2D array) can be provided to
    indicate how to split the 2D array into multiple arrays (each array will contain rows identified by the same
    value in the time points collection).
    :param arr: a 2D array-like object.
    :param time_points: a time points DataFrame.
    :return: a 3D numpy array.
    """
    # Check 2D array
    if not isinstance(arr, (np.ndarray, pd.DataFrame)):
        raise VTypeError(f"Type '{type(arr)}' is not allowed for conversion to 3D array.")

    elif isinstance(arr, pd.DataFrame):
        arr = np.array(arr)

    # Check time points
    if time_points is None:
        time_points = ['0']

    return np.array([arr[time_list == eval(TP)] for TP in time_points], dtype=object)
