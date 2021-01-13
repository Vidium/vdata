# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Collection, Optional

from ..NameUtils import PreSlicer, ArrayLike_2D
from .._IO.errors import VTypeError
from .._IO.logger import generalLogger


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


def reshape_to_3D(arr: ArrayLike_2D, time_points: Optional[Collection[str]], time_list: Optional[Collection[str]]) -> \
        np.ndarray:
    """
    Reshape a 2D array-like object into a 3D array-like. Pandas DataFrames are first converted into numpy arrays.
    By default, 2D arrays of shapes (n, m) are converted to 3D arrays of shapes (1, n, m). In this case, all the data
        in the 2D array is considered to belong to the same time point.

    Optionally, a collection of existing time points can be provided as the 'time_points' parameter. In this
    collection, each time must be given only once. It will be used to split the 2D array into multiple arrays to
    store in the final 3D array.
    In addition to that, a collection of time points of the same length as the number of rows in the 2D array can be
    given as the 'time_list' parameter to describe the 2D array. For each time point in 'time_points',
    rows that match that time point in 'time_list' will be extracted from the 2D array to form a sub-array in the
    final 3D array.
    The 3D array will have shape (T, [n_1, ..., n_T], m) with :
        * T, the number of unique time points in 'time_points'
        * n_i, the number of rows in the sub-array for the time point i
        * m, the number of columns

    :param arr: a 2D array-like object.
    :param time_points: a collection of existing and unique time points.
    :param time_list: a collection of time points describing the rows of the 2D array
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

        if time_list is not None:
            generalLogger.warning("'time_list' parameter provided to reshape_to_3D() without specifying "
                                  "'time_points', it will be ignored.")

            time_list = np.zeros(len(arr))

    else:
        time_points = np.unique(time_points)

        if time_list is None:
            generalLogger.warning("'time_points' parameter provided to reshape_to_3D() without specifying "
                                  "'time_list', all data is considered to come from the same time point.")

    if time_list is None:
        time_list = np.zeros(len(arr))

    return np.array([arr[time_list == eval(str(TP))] for TP in time_points], dtype=object)
