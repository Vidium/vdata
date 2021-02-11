# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Union, Collection, Dict, Optional

import vdata
from .arrays import VObspArrayContainer
from . import views


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


def expand_obsp(data: Optional[Dict[str, pd.DataFrame]], time_points: Dict['vdata.TimePoint', pd.Index]) \
        -> Dict[str, Dict['vdata.TimePoint', pd.DataFrame]]:
    """
    TODO
    """
    if data is None:
        return {}

    _obsp = {}
    for key, DF in data.items():
        _obsp[key] = {}

        for tp, index in time_points.items():
            _obsp[key][tp] = data[key].loc[index, index]

    return _obsp


def compact_obsp(obsp: Union[VObspArrayContainer, 'views.ViewVObspArrayContainer'], index: pd.Index) \
        -> Dict[str, pd.DataFrame]:
    """
    TODO
    """
    _obsp = {key: pd.DataFrame(index=index, columns=index) for key in obsp.keys()}

    index_cumul = 0
    for key in obsp.keys():
        for arr in obsp[key]:
            _obsp[key].iloc[index_cumul:index_cumul + len(arr), index_cumul:index_cumul + len(arr)] = arr
            index_cumul += len(arr)

    return _obsp
