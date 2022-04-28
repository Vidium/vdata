# coding: utf-8
# Created on 30/03/2022 16:28
# Author : matteo

# ====================================================
# imports
import numpy as np
from h5py import string_dtype

from typing import TYPE_CHECKING, Union

from .name_utils import H5Data

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame
    from .view import ViewTemporalDataFrame


# ====================================================
# code
def write_array(array: np.ndarray,
                file: H5Data,
                key: str) -> None:
    """
    Write a numpy array to a H5 file.
    """
    if np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype('bool'):
        dtype = array.dtype

    else:
        dtype = string_dtype()
        array = array.astype(str).astype('O')

    if key in file.keys():
        file[key].resize((len(array),))
        file[key].astype(dtype)
        file[key][()] = array

    else:
        file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None,))


def write_array_chunked(array: np.ndarray,
                        file: H5Data,
                        key: str) -> None:
    """
    Write a numpy array to a H5 file to create a chunked dataset.
    """
    if np.issubdtype(array.dtype, np.number) or array.dtype == np.dtype('bool'):
        dtype = array.dtype

    else:
        dtype = string_dtype()
        array = array.astype(str).astype('O')

    if key in file.keys():
        if file[key].chunks is None:
            del file[key]
            file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None, None))

        else:
            file[key].resize(array.shape)
            file[key].astype(dtype)
            file[key][()] = array

    else:
        file.create_dataset(key, data=array, dtype=dtype, chunks=True, maxshape=(None, None))


def write_TDF(TDF: Union['TemporalDataFrame', 'ViewTemporalDataFrame'],
              file: H5Data) -> None:
    """
    Write a TemporalDataFrame to a H5 file.

    Args:
        TDF: A TemporalDataFrame to write.
        file: A H5 File or Group in which to save the TemporalDataFrame.
    """
    # save attributes
    file.attrs['type'] = 'TDF'
    file.attrs['name'] = TDF.name
    file.attrs['locked_indices'] = TDF.has_locked_indices
    file.attrs['locked_columns'] = TDF.has_locked_columns
    file.attrs['repeating_index'] = TDF.has_repeating_index
    file.attrs['timepoints_column_name'] = "__TDF_None__" if TDF.timepoints_column_name is None else \
        TDF.timepoints_column_name

    # save index
    write_array(TDF.index, file, 'index')

    # save columns numerical
    write_array(TDF.columns_num, file, 'columns_numerical')

    # save columns string
    write_array(TDF.columns_str, file, 'columns_string')

    # save timepoints data
    write_array(TDF.timepoints_column_str, file, 'timepoints')

    # save numerical data
    write_array_chunked(TDF.values_num, file, 'values_numerical')

    # save string data
    write_array_chunked(TDF.values_str, file, 'values_string')
