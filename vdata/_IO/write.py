# coding: utf-8
# Created on 11/20/20 4:30 PM
# Author : matteo

# ====================================================
# imports
import os
import h5py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from functools import singledispatch
from typing import Dict, List, Union
from typing_extensions import Literal

import vdata
from .utils import parse_path
from .logger import generalLogger, getLoggingLevel
from .errors import VPathError
from ..NameUtils import H5Group
from .. import _TDF


# ====================================================
# code
def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


def write_vdata(obj: 'vdata.VData', file: Union[str, Path]) -> None:
    """
    Save this VData object in HDF5 file format.

    :param obj: VData object to save into a .h5 file.
    :param file: path to save the VData.
    """
    file = parse_path(file)

    # make sure the path exists
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

    with h5py.File(file, 'w') as save_file:
        # save layers
        write_data(obj.layers.data, save_file, 'layers')
        # save obs
        write_data(obj.obs, save_file, 'obs')
        # TODO
        # write_data(obj.obsm.data, save_file, 'obsm')
        # write_data(obj.obsp.data, save_file, 'obsp')
        # save var
        write_data(obj.var, save_file, 'var')
        # TODO
        # write_data(obj.varm.data, save_file, 'varm')
        # write_data(obj.varp.data, save_file, 'varp')
        # save time points
        write_data(obj.time_points, save_file, 'time_points')
        # save uns
        write_data(obj.uns, save_file, 'uns')
        # save descriptive data about the VData object
        write_data(obj.dtype, save_file, 'dtype')


def write_vdata_to_csv(obj: 'vdata.VData', directory: Union[str, Path], sep: str = ",", na_rep: str = "",
                       index: bool = True, header: bool = True) -> None:
    """
    Save a VData object into csv files in a directory.

    :param obj: a VData object to save into csv files.
    :param directory: path to a directory for saving the matrices
    :param sep: delimiter character
    :param na_rep: string to replace NAs
    :param index: write row names ?
    :param header: Write col names ?
    """
    directory = parse_path(directory)

    # make sure the directory exists and is empty
    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(os.listdir(directory)):
        raise VPathError("The directory is not empty.")

    # save metadata
    with open(directory / ".metadata.json", 'w') as metadata:
        json.dump({"obs": {"time_points_column_name": obj.obs.time_points_column_name}}, metadata)

    # save matrices
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame obs")
    obj.obs.to_csv(directory / "obs.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame var")
    obj.var.to_csv(directory / "var.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(f"{spacer(1)}Saving TemporalDataFrame time_points")
    obj.time_points.to_csv(directory / "time_points.csv", sep, na_rep, index=index, header=header)

    # TODO
    for dataset in (obj.layers,):  # (obj.obsm, obj.obsp, obj.varm, obj.varp):
        generalLogger.info(f"{spacer(1)}Saving {dataset.name}")
        dataset.to_csv(directory, sep, na_rep, index, header, spacer=spacer(2))

    if obj.uns is not None:
        generalLogger.warning(f"'uns' data stored in VData '{obj.name}' cannot be saved to a csv.")


@singledispatch
def write_data(data, group: H5Group, key: str, key_level: int = 0,
               log_func: Literal['debug', 'info'] = 'info') -> None:
    """
    This is the default function called for writing data to an h5 file.
    Using singledispatch, the correct write_<type> function is called depending on the type of the 'data' parameter.
    If no write_<type> implementation is found, this function defaults and raises a Warning indicating that the
    data could not be saved.

    :param data: data to write.
    :param group: an h5py Group or File to write into.
    :param key: a string for identifying the data.
    :param key_level: for logging purposes, the recursion depth of calls to write_data.
    :param log_func: name of the logging level function to use. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(key_level)}Saving object {key}")
    generalLogger.warning(f"H5 writing not yet implemented for data of type '{type(data)}'.")


@write_data.register(dict)
def write_Dict(data: Dict, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing dictionaries to the h5 file.
    It creates a group for storing the keys and recursively calls write_data to store them.
    """
    generalLogger.info(f"{spacer(key_level)}Saving dict {key}")
    grp = group.create_group(str(key))
    grp.attrs['type'] = 'dict'

    for dict_key, value in data.items():
        write_data(value, grp, dict_key, key_level=key_level+1)


@write_data.register
def write_DataFrame(data: pd.DataFrame, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing pd.DataFrames to the h5 file. Each DataFrame is stored in a group, containing the index and the
    columns as Series.
    Used for obs, var, time_points.
    """
    generalLogger.info(f"{spacer(key_level)}Saving DataFrame {key}")

    df_group = group.create_group(str(key))
    df_group.attrs['type'] = 'DF'

    # save column order
    df_group.attrs["column_order"] = list(data.columns)

    # save index and series
    df_group.attrs["index"] = list(data.index)

    for col_name, series in data.items():
        write_data(series, df_group, str(col_name), key_level=key_level+1)


@write_data.register
def write_TemporalDataFrame(data: '_TDF.TemporalDataFrame', group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing TemporalDataFrames to the h5 file. Each TemporalDataFrame is stored in a group, containing the
    index and the columns as Series.
    Used for layers, obs, obsm.
    """
    generalLogger.info(f"{spacer(key_level)}Saving TemporalDataFrame {key}")

    df_group = group.create_group(str(key))
    df_group.attrs['type'] = 'TDF'

    # save column order
    df_group.attrs["column_order"] = list(data.columns)

    # save index and series
    df_group.attrs["index"] = list(data.index)
    write_data(data.time_points_column_name, df_group, 'time_col', key_level=key_level + 1)
    write_data(data.time_points_column, df_group, 'time_list', key_level=key_level + 1)

    log_func: Literal['debug', 'info'] = 'info'

    for i, (col_name, series) in enumerate(data.items()):
        write_series(series, df_group, str(col_name), key_level=key_level + 1, log_func=log_func)

        if log_func == 'info' and i > 0:
            log_func = 'debug'

    if getLoggingLevel() != 'DEBUG':
        generalLogger.info(f"{spacer(key_level+1)}...")


@write_data.register(pd.Series)
@write_data.register(pd.Index)
def write_series(series: pd.Series, group: H5Group, key: str, key_level: int = 0,
                 log_func: Literal['debug', 'info'] = 'info') -> None:
    """
    Function for writing pd.Series to the h5 file. The Series are expected to belong to a group (a DataFrame or in uns).
    """
    getattr(generalLogger, log_func)(f"{spacer(key_level)}Saving Series {key}")

    # Series of strings
    if series.dtype == object:
        group.create_dataset(str(key), data=series.values, dtype=h5py.string_dtype(encoding='utf-8'))

    # Series of categorical data
    elif pd.api.types.is_categorical_dtype(series):
        series_group = group.create_group(str(key))
        # save values
        values = pd.Series(np.array(series.values))
        write_data(values, series_group, "values", key_level=key_level+1, log_func=log_func)
        # save categories
        series_group.attrs["categories"] = np.array(series.values.categories, dtype='S')
        # save ordered
        series_group.attrs["ordered"] = series.values.ordered

    # Series of regular data
    else:
        group[str(key)] = series.values

    group[str(key)].attrs['type'] = 'series'


@write_data.register
def write_array(data: np.ndarray, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing np.arrays to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving array {key}")
    if data.dtype.type == np.str_:
        group.create_dataset(str(key), data=data.astype('S'))
    else:
        print(data, type(data))
        group[str(key)] = data

    group[str(key)].attrs['type'] = 'array'


@write_data.register(list)
def write_list(data: List, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing lists to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving list {key}")
    write_data(np.array(data), group, key, key_level=key_level+1)


@write_data.register(str)
@write_data.register(np.str_)
@write_data.register(int)
@write_data.register(np.integer)
@write_data.register(float)
@write_data.register(np.floating)
@write_data.register(bool)
@write_data.register(np.bool_)
def write_single_value(data: Union[str, np.str_, int, np.integer, float, np.floating, bool, np.bool_],
                       group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing a single value to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving single value {key}")
    group[str(key)] = data
    group[str(key)].attrs['type'] = 'value'


@write_data.register
def write_Type(data: type, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing a type to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving type {key}")
    group[str(key)] = data.__name__
    group[str(key)].attrs['type'] = 'type'


@write_data.register
def write_None(_: None, group: H5Group, key: str, key_level: int = 0) -> None:
    """
    Function for writing None to the h5 file.
    """
    generalLogger.info(f"{spacer(key_level)}Saving None value for {key}")
    _ = group.create_group(str(key))
    group[str(key)].attrs['type'] = 'None'
