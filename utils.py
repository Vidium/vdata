# coding: utf-8
# Created on 11/16/20 11:35 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import scipy.sparse as sp
from typing import Union, List


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


def is_in(obj: np.ndarray, list_arrays: Union[np.ndarray, List[np.ndarray]]):
    for arr in list_arrays:
        if are_equal(obj, arr):
            return True
        else:
            return False
