# coding: utf-8
# Created on 11/20/20 4:30 PM
# Author : matteo

# ====================================================
# imports
import h5py
import pandas as pd
import numpy as np
from functools import singledispatch
from typing import Dict, List, Union

from .logger import generalLogger

H5Group = Union[h5py.File, h5py.Group]


# ====================================================
# code
@singledispatch
def write_data(data, group: H5Group, key: str) -> None:
    """
    This is the default function called for writing data to an h5 file.
    Using singledispatch, the correct write_<type> function is called depending on the type of the 'data' parameter.
    If no write_<type> implementation is found, this function defaults and raises a Warning indicating that the
    data could not be saved.

    :param data: data to write
    :param group: an h5py Group or File to write into
    :param key: a string for identifying the data
    """
    generalLogger.warning(f"H5 writing not yet implemented for data of type '{type(data)}'.")


@write_data.register(dict)
def write_Dict(data: Dict, group: H5Group, key: str) -> None:
    """
    Function for writing dictionaries to the h5 file.
    It creates a group for storing the keys and recursively calls write_data to store them.
    """
    generalLogger.info(f"Saving dict {key}")
    grp = group.create_group(key)
    for dict_key, value in data.items():
        write_data(value, grp, dict_key)


@write_data.register
def write_DataFrame(data: pd.DataFrame, group: H5Group, key: str) -> None:
    """
    Function for writing pd.DataFrames to the h5 file. Each DataFrame is stored in a group, containing the index and the
    columns as Series.
    Used for obs, var, time_points.
    """
    generalLogger.info(f"Saving DataFrame {key}")
    # save attributes
    df_group = group.create_group(key)
    df_group.attrs["column_order"] = list(data.columns)

    # save index and series
    df_group.attrs["index"] = list(data.index)

    for col_name, series in data.items():
        write_data(series, df_group, str(col_name))


@write_data.register(pd.Series)
@write_data.register(pd.Index)
def write_series(series: pd.Series, group: H5Group, key: str) -> None:
    """
    Function for writing pd.Series to the h5 file. The Series are expected to belong to a group (a DataFrame or in uns).
    """
    generalLogger.info(f"Saving Series {key}")
    # Series of strings
    if series.dtype == object:
        group.create_dataset(key, data=series.values, dtype=h5py.string_dtype(encoding='utf-8'))

    # Series of categorical data
    elif pd.api.types.is_categorical_dtype(series):
        series_group = group.create_group(key)
        # save values
        values = pd.Series(np.array(series.values))
        write_data(values, series_group, "values")
        # save categories
        series_group.attrs["categories"] = np.array(series.values.categories, dtype='S')
        # save ordered
        series_group.attrs["ordered"] = series.values.ordered

    # Series of regular data
    else:
        group[key] = series.values


@write_data.register
def write_array(data: np.ndarray, group: H5Group, key: str) -> None:
    """
    Function for writing np.arrays to the h5 file.
    """
    generalLogger.info(f"Saving array {key}")
    if data.dtype.type == np.str_:
        group.create_dataset(key, data=data.astype('S'))
    else:
        group[key] = data


@write_data.register(list)
def write_list(data: List, group: H5Group, key: str) -> None:
    """
    Function for writing lists to the h5 file.
    """
    generalLogger.info(f"Saving list {key}")
    write_data(np.array(data), group, key)


@write_data.register(str)
@write_data.register(np.str_)
@write_data.register(int)
@write_data.register(np.integer)
@write_data.register(float)
@write_data.register(np.floating)
@write_data.register(bool)
@write_data.register(np.bool_)
def write_single_value(data: Union[str, np.str_, int, np.integer, float, np.floating, bool, np.bool_], group: H5Group, key: str) -> None:
    """
    Function for writing a single value to the h5 file.
    """
    generalLogger.info(f"Saving single value {key}")
    group[key] = data


@write_data.register
def write_None(_: None, group: H5Group, key: str) -> None:
    """
    Function for writing None to the h5 file.
    """
    generalLogger.info(f"Saving None value for {key}")
    _ = group.create_group(key)
