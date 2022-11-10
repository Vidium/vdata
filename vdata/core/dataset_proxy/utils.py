# coding: utf-8
# Created on 22/10/2022 15:35
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from vdata.h5pickle import Dataset

from vdata.time_point import TimePoint
from vdata.core.dataset_proxy.base import BaseDatasetProxy, DATASET_DTYPE
from vdata.core.dataset_proxy.dataset_1D import NumDatasetProxy1D, StrDatasetProxy1D, TPDatasetProxy1D
from vdata.core.dataset_proxy.dataset_2D import NumDatasetProxy2D, StrDatasetProxy2D


# ====================================================
# code
def is_str_dtype(dtype) -> bool:
    if dtype == object or dtype.type == np.bytes_:
        return True

    return False


def auto_DatasetProxy(dataset: Dataset,
                      view_on: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
                      dtype: DATASET_DTYPE | None = None) -> BaseDatasetProxy | np.ndarray:
    """
    Get a DatasetProxy of the correct type for the dataset.
    /!\ Works only for numeric and string datasets, datasets of custom objects are not handled.

    Args:
        dataset: a h5 Dataset object for which to build a DatasetProxy.
        view_on: define a view on the h5 Dataset (one array for 1D datasets, 2 arrays for 2D datasets).
        dtype: a data type to cast the Dataset.
    """
    # fix a data type
    if dtype is not None and not np.issubdtype(dtype, np.generic) and not dtype == TimePoint:
        raise TypeError(f"Type '{type(dtype)}' not recognized for a data type.")

    if dtype is None:
        if is_str_dtype(dataset.dtype):
            dtype = np.str_

        else:
            dtype = dataset.dtype

    if dataset.ndim == 0:
        raise TypeError('Datasets of dimension 0 are not handled.')

    # create a dataset proxy of the correct type
    elif dataset.ndim == 1:
        if is_str_dtype(dataset.dtype):
            if np.issubdtype(dtype, np.number):
                raise NotImplementedError('Conversion (str --> num) not supported yet.')

            elif np.issubdtype(dtype, np.str_):
                return StrDatasetProxy1D(dataset, view_on)

            elif dtype == TimePoint:
                return TPDatasetProxy1D(dataset, view_on)

            else:
                raise TypeError

        else:
            if np.issubdtype(dtype, np.number):
                return NumDatasetProxy1D(dataset, view_on)

            elif np.issubdtype(dtype, np.str_):
                raise NotImplementedError('Conversion (num --> str) not supported yet.')

            elif dtype == TimePoint:
                raise NotImplementedError('Conversion (num --> tp) not supported yet.')

            else:
                raise TypeError

    elif dataset.ndim == 2:
        if is_str_dtype(dataset.dtype):
            if np.issubdtype(dtype, np.number):
                raise NotImplementedError('Conversion (str --> num) not supported yet.')

            elif np.issubdtype(dtype, np.str_):
                return StrDatasetProxy2D(dataset, view_on)

            else:
                raise TypeError

        else:
            if np.issubdtype(dtype, np.number):
                return NumDatasetProxy2D(dataset, view_on)

            elif np.issubdtype(dtype, np.str_):
                raise NotImplementedError('Conversion (num --> str) not supported yet.')

            else:
                raise TypeError

    else:
        # FIXME : temporary solution for 3+ dimensional arrays: load them all in RAM as numpy arrays
        return dataset[:]
