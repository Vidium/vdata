# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
from typing import Any, Collection

import numpy as np
import numpy.typing as npt


# ====================================================
# code
# Identification ---------------------------------------------------------
def array_isin(array: npt.NDArray[Any], 
               list_arrays: npt.NDArray[Any] | Collection[npt.NDArray[Any]]) -> bool:
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
