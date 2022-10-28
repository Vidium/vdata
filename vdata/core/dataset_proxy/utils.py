# coding: utf-8
# Created on 22/10/2022 15:35
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from h5py import Dataset

from vdata.core.dataset_proxy.base import BaseDatasetProxy
from vdata.core.dataset_proxy.dataset_1D import NumDatasetProxy1D, StrDatasetProxy1D, TPDatasetProxy1D
from vdata.core.dataset_proxy.dataset_2D import NumDatasetProxy2D, StrDatasetProxy2D
from vdata.core.dataset_proxy.dtypes import DType, num_, int_, float_, str_, tp_, issubdtype


# ====================================================
# code
def is_str_dtype(dtype) -> bool:
    if dtype == object or dtype.type == np.bytes_:
        return True

    return False


def auto_DatasetProxy(dataset: Dataset,
                      view_on: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
                      dtype: DType | None = None) -> BaseDatasetProxy:
    """
    Get a DatasetProxy of the correct type for the dataset.
    /!\ Works only for numeric and string datasets, datasets of custom objects are not handled.

    Args:
        dataset: a h5 Dataset object for which to build a DatasetProxy.
        view_on: define a view on the h5 Dataset (one array for 1D datasets, 2 arrays for 2D datasets).
        dtype: a data type to cast the Dataset.
    """
    # fix a data type
    if dtype is not None and not isinstance(dtype, DType):
        raise TypeError(f"Type '{type(dtype)}' not recognized for a data type.")

    if dtype is None:
        if is_str_dtype(dataset.dtype):
            dtype = str_

        else:
            dtype = num_

    if dtype == num_:
        if np.issubdtype(dataset.dtype, int):
            dtype = int_

        elif np.issubdtype(dataset.dtype, float):
            dtype = float_

        else:
            raise TypeError

    # create a dataset proxy of the correct type
    if dataset.ndim == 1:
        if is_str_dtype(dataset.dtype):
            if issubdtype(dtype, num_):
                raise NotImplementedError('Conversion (str --> num) not supported yet.')

            elif dtype == str_:
                return StrDatasetProxy1D(dataset, view_on)

            elif dtype == tp_:
                return TPDatasetProxy1D(dataset, view_on)

            else:
                raise TypeError

        else:
            if issubdtype(dtype, num_):
                return NumDatasetProxy1D(dataset, view_on)

            elif dtype == str_:
                raise NotImplementedError('Conversion (num --> str) not supported yet.')

            elif dtype == tp_:
                raise NotImplementedError('Conversion (num --> tp) not supported yet.')

            else:
                raise TypeError

    elif dataset.ndim == 2:
        if is_str_dtype(dataset.dtype):
            if issubdtype(dtype, num_):
                raise NotImplementedError('Conversion (str --> num) not supported yet.')

            elif dtype == str_:
                return StrDatasetProxy2D(dataset, view_on)

            else:
                raise TypeError

        else:
            if issubdtype(dtype, num_):
                return NumDatasetProxy2D(dataset, view_on)

            elif dtype == str_:
                raise NotImplementedError('Conversion (num --> str) not supported yet.')

            else:
                raise TypeError

    raise TypeError('Datasets of dimension 0 or greater than 2 are not handled.')
