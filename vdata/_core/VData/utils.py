# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Union, Collection, Optional

from vdata.time_point import TimePoint


# ====================================================
# code
# Identification ---------------------------------------------------------
def array_isin(array: np.ndarray, list_arrays: Union[np.ndarray, Collection[np.ndarray]]) -> bool:
    """
    Whether a given array is in a collection of arrays.
    :param array: an array.
    :param list_arrays: a collection of arrays.
    :return: whether the array is in the collection of arrays.
    """
    for target_array in list_arrays:
        if np.array_equal(array, target_array):
            return True

    return False


# Formatting & Conversion ------------------------------------------------
def expand_obsp(data: Optional[dict[str, pd.DataFrame]],
                time_points: dict['TimePoint', pd.Index]) \
        -> dict[str, dict['TimePoint', pd.DataFrame]]:
    """
    Transform square pandas DataFrames describing an obsp into multiple smaller square pandas DataFrames by cutting
    by TimePoint.
    :param data: an optional dictionary of str:square pandas DataFrames.
    :param time_points: a dictionary of TimePoint:index at TimePoint.
    :return: a dictionary of str:dictionary of TimePoint:square DataFrame for the TimePoint.
    """
    if data is None:
        return {}

    return {key: {tp: data[key].loc[index, index] for tp, index in time_points.items()} for key, DF in data.items()}
