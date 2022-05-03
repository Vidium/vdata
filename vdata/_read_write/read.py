# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
import os
import json
import shutil
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from typing import Union, Optional, Any, Callable, Collection, cast, Literal

from numpy import int8, int16, int32, int64, float16, float32, float64, float128  # noqa: F401

import vdata
from .utils import parse_path, H5GroupReader
from .._core.TDF import read_TDF, read_TemporalDataFrame_from_csv
from ..name_utils import DType
from ..utils import get_value, repr_array
from ..vdataframe import VDataFrame
from ..time_point import TimePoint
from ..IO import generalLogger, VValueError, VTypeError, ShapeError
from .._core import TemporalDataFrame
from ..h5pickle import File, Dataset, Group


def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


# ====================================================
# code
# CSV file format ---------------------------------------------------------------------------------
def read_from_csv(directory: Union[Path, str],
                  dtype: 'DType' = np.float32,
                  time_list: Optional[Union[Collection, 'DType', Literal['*']]] = None,
                  time_col: Optional[str] = None,
                  name: Optional[Any] = None) -> 'vdata.VData':
    """
    Function for reading data from csv datasets and building a VData object.

    Args:
        directory: a path to a directory containing csv datasets.
            The directory should have the format, for any combination of the following datasets :
                ⊦ layers
                    ⊦ <...>.csv
                ⊦ obsm
                    ⊦ <...>.csv
                ⊦ obsp
                    ⊦ <...>.csv
                ⊦ varm
                    ⊦ <...>.csv
                ⊦ varp
                    ⊦ <...>.csv
                ⊦ obs.csv
                ⊦ timepoints.csv
                ⊦ var.csv
        dtype: data type to force on the newly built VData object.
        time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
        time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
            given. This column will be used as the time data.
        name: an optional name for the loaded VData object.
    """
    directory = parse_path(directory)

    # make sure the path exists
    if not os.path.exists(directory):
        raise VValueError(f"The path {directory} does not exist.")

    # load metadata if possible
    metadata = None

    if os.path.isfile(directory / ".metadata.json"):
        with open(directory / ".metadata.json", 'r') as metadata_file:
            metadata = json.load(metadata_file)

    data = {'obs': None, 'var': None, 'timepoints': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None, 'dtype': dtype}

    # import the data
    for f in os.listdir(directory):
        if f != ".metadata.json":
            generalLogger.info(f"Got key : '{f}'.")

            if f.endswith('.csv'):
                if f in ('var.csv', 'timepoints.csv'):
                    generalLogger.info(f"{spacer(1)}Reading pandas DataFrame '{f[:-4]}'.")
                    data[f[:-4]] = pd.read_csv(directory / f, index_col=0)

                elif f in ('obs.csv', ):
                    generalLogger.info(f"{spacer(1)}Reading TemporalDataFrame '{f[:-4]}'.")
                    if time_list is None and time_col is None:
                        if metadata is not None:
                            this_time_col = metadata['obs']['timepoints_column_name']
                        else:
                            this_time_col = None
                    else:
                        this_time_col = time_col

                    data[f[:-4]] = read_TemporalDataFrame_from_csv(directory / f,
                                                                   time_list=time_list,
                                                                   time_col_name=this_time_col)

            else:
                dataset_dict = {}
                generalLogger.info(f"{spacer(1)}Reading group '{f}'.")

                for dataset in os.listdir(directory / f):
                    if f in ('layers', 'obsm'):
                        generalLogger.info(f"{spacer(2)} Reading TemporalDataFrame {dataset[:-4]}")
                        if time_list is None and time_col is None:
                            if metadata is not None:
                                this_time_col = metadata[f][dataset[:-4]]['timepoints_column_name']
                            else:
                                this_time_col = None
                        else:
                            this_time_col = time_col

                        dataset_dict[dataset[:-4]] = read_TemporalDataFrame_from_csv(directory / f / dataset,
                                                                                     time_list=time_list,
                                                                                     time_col_name=this_time_col)

                    elif f in ('varm', 'varp'):
                        generalLogger.info(f"{spacer(2)} Reading pandas DataFrame {dataset}")
                        dataset_dict[dataset[:-4]] = pd.read_csv(directory / f, index_col=0)

                    else:
                        raise NotImplementedError

                data[f] = dataset_dict

    return vdata.VData(data['layers'],
                       data['obs'],
                       data['obsm'],
                       data['obsp'],
                       data['var'],
                       data['varm'],
                       data['varp'],
                       data['timepoints'],
                       dtype=data['dtype'],
                       name=name)


# GPU output --------------------------------------------------------------------------------------
def read_from_dict(data: dict[str, dict[Union['DType', str], Union[np.ndarray, pd.DataFrame]]],
                   obs: Optional[Union[pd.DataFrame, 'TemporalDataFrame']] = None,
                   var: Optional[pd.DataFrame] = None,
                   timepoints: Optional[pd.DataFrame] = None,
                   dtype: Optional['DType'] = None,
                   name: Optional[Any] = None) -> 'vdata.VData':
    """
    Load a simulation's recorded information into a VData object.

    If time points are not given explicitly, this function will try to recover them from the time point names in
        the data.
    For this to work, time points names must be strings with :
        - last character in (s, m, h, D, M, Y)
        - first characters convertible to a float
    The last character indicates the unit:
        - s : seconds
        - m : minutes
        - h : hours
        - D : days
        - M : months
        - Y : years

    :param data: a dictionary of data types (RNA, Proteins, etc.) linked to dictionaries of time points linked to
        matrices of cells x genes.
    :param obs: a pandas DataFrame describing the observations (cells).
    :param var: a pandas DataFrame describing the variables (genes).
    :param timepoints: a pandas DataFrame describing the time points.
    :param dtype: the data type for the matrices in VData.
    :param name: an optional name for the loaded VData object.

    :return: a VData object containing the simulation's data
    """
    _data = {}
    _timepoints: list[TimePoint] = []
    check_tp = False

    _index = obs.index if isinstance(obs, (pd.DataFrame, TemporalDataFrame)) else None
    generalLogger.debug(f"Found index is : {repr_array(_index)}.")

    _columns = var.index if isinstance(var, pd.DataFrame) else None
    generalLogger.debug(f"Found columns is : {repr_array(_columns)}.")

    if not isinstance(data, dict):
        raise VTypeError("Data should be a dictionary with format : {data type: {time point: matrix}}")

    else:
        for data_index, (data_type, TP_matrices) in enumerate(data.items()):
            if not isinstance(TP_matrices, dict):
                raise VTypeError(f"'{data_type}' in data should be a dictionary with format : {{time point: matrix}}")

            # ---------------------------------------------------------------------------
            generalLogger.debug(f"Loading layer '{data_type}'.")

            for matrix_index, matrix in TP_matrices.items():
                matrix_TP = TimePoint(matrix_index)

                if not isinstance(matrix, (np.ndarray, pd.DataFrame)) or matrix.ndim != 2:
                    raise VTypeError(f"Item at time point '{matrix_TP}' is not a 2D array-like object "
                                     f"(numpy ndarray, pandas DatFrame).")

                elif check_tp:
                    if matrix_TP not in _timepoints:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _timepoints.append(matrix_TP)

            check_tp = True

            _layer_data = np.vstack(list(TP_matrices.values()))
            # check data matches columns shape
            if _columns is not None and _layer_data.shape[1] != len(_columns):
                raise ShapeError(f"Layer '{data_type}' has {_layer_data.shape[1]} columns , should have "
                                 f"{len(_columns)}.")

            _loaded_data = pd.DataFrame(np.vstack(list(TP_matrices.values())), columns=_columns)

            _time_list = [_timepoints[matrix_index] for matrix_index, matrix in enumerate(TP_matrices.values())
                          for _ in range(len(matrix))]
            generalLogger.debug(f"Computed time list to be : {repr_array(_time_list)}.")

            _data[data_type] = TemporalDataFrame(data=_loaded_data,
                                                 # timepoints=_timepoints,
                                                 time_list=_time_list,
                                                 index=_index)

            generalLogger.info(f"Loaded layer '{data_type}' ({data_index+1}/{len(data)})")

        # if time points is not given, build a DataFrame from time points found in 'data'
        if timepoints is None:
            timepoints = pd.DataFrame({"value": _timepoints})

        return vdata.VData(_data, obs=obs, var=var, timepoints=timepoints, dtype=dtype, name=name)


# HDF5 file format --------------------------------------------------------------------------------
def read(file: Union[Path, str], mode: Literal['r', 'r+'] = 'r',
         dtype: Optional['DType'] = None,
         name: Optional[Any] = None,
         backup: bool = False) -> 'vdata.VData':
    """
    Function for reading data from an h5 file and building a VData object from it.

    Args:
        file: path to an h5 file.
        mode: reading mode : 'r' (read only) or 'r+' (read and write).
        dtype: data type to force on the newly built VData object. If set to None, the dtype is inferred from
            the h5 file.
        name: an optional name for the loaded VData object.
        backup: create a backup copy of the read h5 file in case something goes wrong ?
    """
    generalLogger.debug("\u23BE read VData : begin -------------------------------------------------------- ")
    file = parse_path(file)

    if file.suffix != '.vd':
        raise VValueError("Cannot read file with suffix != '.vd'.")

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    if backup:
        backup_file_suffix = '.backup'
        backup_nb = 0

        while os.path.exists(str(file) + backup_file_suffix):
            backup_nb += 1
            backup_file_suffix = f"({backup_nb}).backup"

        shutil.copy(file, str(file) + backup_file_suffix)
        generalLogger.info(f"Backup file '{str(file) + backup_file_suffix}' was created.")

    data = {'obs': None, 'var': None, 'timepoints': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None,
            'uns': {}}

    # import data from file
    importFile = H5GroupReader(File(str(file), mode))
    for key in importFile.keys():
        generalLogger.info(f"Got key : '{key}'.")

        if key in ('obs', 'var', 'timepoints', 'layers', 'obsm', 'obsp', 'varm', 'varp', 'uns'):
            type_ = importFile[key].attrs('type')
            data[key] = func_[type_](importFile[key], 1, mode)

        else:
            generalLogger.warning(f"Unexpected data with key {key} while reading file, skipping.")

    data['timepoints']['value'] = [TimePoint(tp) for tp in data['timepoints']['value']]

    new_VData = vdata.VData(data['layers'],
                            data['obs'], data['obsm'], data['obsp'],
                            data['var'], data['varm'], data['varp'],
                            data['timepoints'], data['uns'], dtype=dtype,
                            name=name, file=importFile,
                            no_check=True)

    # if data['obsp'] is not None:
    #     for key, arr in data['obsp'].items():
    #         new_VData.obsp[key] = arr

    generalLogger.debug("\u23BF read VData : end -------------------------------------------------------- ")

    return new_VData


# from HDF5 groups --------------------------------------------------------------------------------
def read_h5_dict(group: H5GroupReader, level: int = 1, mode: Literal['r', 'r+'] = 'r') -> dict:
    """
    Function for reading a dictionary from an h5 file.

    Args:
        group: a H5GroupReader from which to read a dictionary.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        mode:  mode for reading the TDF.
    """
    generalLogger.info(f"{spacer(level)}Reading dict {group.name}.")

    start = time()
    data = {}

    for dataset_key in group.keys():
        dataset_type = cast(H5GroupReader, group[dataset_key]).attrs("type")

        data[get_value(dataset_key)] = func_[dataset_type](group[dataset_key], level=level+1, mode=mode)

    end = time()

    generalLogger.info(f"{spacer(level)}(took {end - start:1.4f} seconds)).")
    return data


def read_h5_VDataFrame(group: H5GroupReader, level: int = 1, *args, **kwargs) -> VDataFrame:
    """
    Function for reading a pandas DataFrame from an h5 file.

    :param group: a H5GroupReader from which to read a DataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading VDataFrame {group.name}.")

    start = time()
    # get index
    dataset_type = cast(H5GroupReader, group['index']).attrs("type")
    index = func_[dataset_type](group['index'], level=level + 1)

    # get columns
    dataset_type = cast(H5GroupReader, group['columns']).attrs("type")
    columns = func_[dataset_type](group['columns'], level=level + 1)

    # get data
    data = None

    if 'data_numeric' in group.keys():
        generalLogger.info(f"{spacer(level + 1)}Reading numeric data.")
        dataset_type = cast(H5GroupReader, group['data_numeric']['columns']).attrs("type")
        data_numeric_columns = func_[dataset_type](group['data_numeric']['columns'], level=level + 2)

        dataset_type = cast(H5GroupReader, group['data_numeric']['data']).attrs("type")
        data_numeric_data = func_[dataset_type](group['data_numeric']['data'], level=level + 2)

        data = pd.DataFrame(data_numeric_data, columns=data_numeric_columns, index=index)

    if 'data_str' in group.keys():
        generalLogger.info(f"{spacer(level + 1)}Reading non numeric data.")
        dataset_type = cast(H5GroupReader, group['data_str']['columns']).attrs("type")
        data_str_columns = func_[dataset_type](group['data_str']['columns'], level=level + 2)

        dataset_type = cast(H5GroupReader, group['data_str']['data']).attrs("type")
        data_str_data = func_[dataset_type](group['data_str']['data'], level=level + 2)

        if data is not None:
            for col_i, col in enumerate(data_str_columns):
                col_index = int(np.where(columns == col)[0][0])
                data.insert(col_index, col, data_str_data[:, col_i])

        else:
            data = pd.DataFrame(data_str_data, columns=data_str_columns, index=index)

    end = time()
    generalLogger.info(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

    return VDataFrame(data=data, index=index, columns=columns, file=group.group)


def read_h5_TemporalDataFrame(group: H5GroupReader, level: int = 1, mode: Literal['r', 'r+'] = 'r') \
        -> 'TemporalDataFrame':
    """
    Function for reading a TemporalDataFrame from an h5 file.

    Args:
        group: a H5GroupReader from which to read a TemporalDataFrame.
        level: for logging purposes, the recursion depth of calls to a read_h5 function.
        mode: mode for reading the TDF.
    """
    return read_TDF(group.group, mode=mode)


def read_h5_series(group: H5GroupReader, index: Optional[list] = None, level: int = 1,
                   log_func: Literal['debug', 'info'] = 'info', *args, **kwargs) -> pd.Series:
    """
    Function for reading a pandas Series from an h5 file.

    :param group: an H5GroupReader from which to read a Series.
    :param index: an optional list representing the indexes for the Series.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    :param log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(level)}Reading Series {group.name}.")

    start = time()
    # simple Series
    if group.isinstance(Dataset):
        data_type = get_dtype_from_string(group.attrs('dtype'))
        if data_type == TimePoint:
            values = [TimePoint(tp, no_check=True) for tp in read_h5_array(group, level=level+1, log_func=log_func)]

        else:
            values = list(map(data_type, read_h5_array(group, level=level+1, log_func=log_func)))

        end = time()
        getattr(generalLogger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return pd.Series(values, index=index)

    # categorical Series
    elif group.isinstance(Group):
        # get data
        data_type = get_dtype_from_string(group.attrs('dtype'))
        categories = read_h5_array(cast(H5GroupReader, group['categories']), level=level+1, log_func=log_func)
        ordered = get_value(group.attrs('ordered'))
        values = list(map(data_type, read_h5_array(cast(H5GroupReader, group['values']), level=level+1,
                                                   log_func=log_func)))

        end = time()
        getattr(generalLogger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return pd.Series(pd.Categorical(values, categories, ordered=ordered), index=index)

    # unexpected type
    else:
        raise VTypeError(f"Unexpected type {type(group)} while reading h5 file.")


def read_h5_array(group: H5GroupReader, level: int = 1,
                  log_func: Literal['debug', 'info'] = 'info', *args, **kwargs) -> np.ndarray:
    """
    Function for reading a numpy array from an h5 file.
    If the imported array contains strings, as they where stored as bytes, they are converted back to strings.

    :param group: a H5GroupReader from which to read an array.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    :param log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(level)}Reading array {group.name}.")

    start = time()
    arr = group[()]

    if isinstance(arr, np.ndarray):
        # fix string arrays (object type to strings)
        if arr.dtype.type is np.object_:
            try:
                end = time()
                getattr(generalLogger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

                return arr.astype(float)

            except ValueError:
                end = time()
                getattr(generalLogger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

                return arr.astype(np.str_)

        end = time()
        getattr(generalLogger, log_func)(f"{spacer(level)}(took {end - start:1.4f} seconds)).")

        return arr

    else:
        raise VTypeError(f"Group is not an array (type is '{type(arr)}').")


def read_h5_value(group: H5GroupReader, level: int = 1, *args, **kwargs) -> Union[str, int, float, bool, type]:
    """
    Function for reading a value from an h5 file.

    :param group: a H5GroupReader from which to read a value.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading value {group.name}.")

    if group.is_string():
        return group.as_string()

    return get_value(group[()])


def read_h5_path(group: H5GroupReader, level: int = 1, *args, **kwargs) -> Path:
    """
    Function for reading a Path from an h5 file.

    :param group: a H5GroupReader from which to read a Path.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading Path {group.name}.")

    return Path(group.as_string())


def read_h5_None(_: H5GroupReader, level: int = 1, *args, **kwargs) -> None:
    """
    Function for reading 'None' from an h5 file.

    :param _: a H5GroupReader from which to read a value.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading None.")
    return None


func_: dict[str, Callable] = {
    'dict': read_h5_dict,
    'VDF': read_h5_VDataFrame,
    'TDF': read_h5_TemporalDataFrame,
    'series': read_h5_series,
    'array': read_h5_array,
    'value': read_h5_value,
    'type': read_h5_value,
    'path': read_h5_path,
    'None': read_h5_None
}


def get_dtype_from_string(dtype_str: str) -> type:
    """
    Parse the given string into a data type.

    :param dtype_str: string to parse.

    :return: a parsed data type.
    """
    if dtype_str.startswith("<class '"):
        return eval(dtype_str[8:-2])

    elif dtype_str in ('object', 'category'):
        return str

    else:
        return eval(dtype_str)
