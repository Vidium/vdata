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
from pathlib import Path
from typing import Union, Optional, Dict, List, Any, Callable, Collection, cast
from typing_extensions import Literal

from numpy import int8, int16, int32, int64, float16, float32, float64, float128  # noqa: F401

import vdata
from .utils import parse_path, H5GroupReader
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
                  time_points: Optional[Collection[str]] = None,
                  name: Optional[Any] = None) -> 'vdata.VData':
    """
    Function for reading data from csv datasets and building a VData object.

    :param directory: a path to a directory containing csv datasets.
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
            ⊦ time_points.csv
            ⊦ var.csv
    :param dtype: data type to force on the newly built VData object.
    :param time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
    :param time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
        given. This column will be used as the time data.
    :param time_points: a list of time points that should exist. This is useful when using the '*' character to
        specify the list of time points that the TemporalDataFrame should cover.
    :param name: an optional name for the loaded VData object.
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

    data = {'obs': None, 'var': None, 'time_points': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None, 'dtype': dtype}

    # import the data
    for f in os.listdir(directory):
        if f != ".metadata.json":
            generalLogger.info(f"Got key : '{f}'.")

            if f.endswith('.csv'):
                if f in ('var.csv', 'time_points.csv'):
                    generalLogger.info(f"{spacer(1)}Reading pandas DataFrame '{f[:-4]}'.")
                    data[f[:-4]] = pd.read_csv(directory / f, index_col=0)

                elif f in ('obs.csv', ):
                    generalLogger.info(f"{spacer(1)}Reading TemporalDataFrame '{f[:-4]}'.")
                    if time_list is None and time_col is None:
                        if metadata is not None:
                            this_time_col = metadata['obs']['time_points_column_name']
                        else:
                            this_time_col = None
                    else:
                        this_time_col = time_col

                    data[f[:-4]] = read_from_csv_TemporalDataFrame(directory / f, time_list=time_list,
                                                                   time_col=this_time_col, time_points=time_points)

            else:
                dataset_dict = {}
                generalLogger.info(f"{spacer(1)}Reading group '{f}'.")

                for dataset in os.listdir(directory / f):
                    if f in ('layers', 'obsm'):
                        generalLogger.info(f"{spacer(2)} Reading TemporalDataFrame {dataset[:-4]}")
                        if time_list is None and time_col is None:
                            if metadata is not None:
                                this_time_col = metadata[f][dataset[:-4]]['time_points_column_name']
                            else:
                                this_time_col = None
                        else:
                            this_time_col = time_col

                        dataset_dict[dataset[:-4]] = read_from_csv_TemporalDataFrame(directory / f / dataset,
                                                                                     time_list=time_list,
                                                                                     time_col=this_time_col,
                                                                                     time_points=time_points)

                    elif f in ('varm', 'varp'):
                        generalLogger.info(f"{spacer(2)} Reading pandas DataFrame {dataset}")
                        dataset_dict[dataset[:-4]] = pd.read_csv(directory / f, index_col=0)

                    else:
                        raise NotImplementedError

                data[f] = dataset_dict

    return vdata.VData(data['layers'],
                       data['obs'], data['obsm'], data['obsp'],
                       data['var'], data['varm'], data['varp'],
                       data['time_points'], dtype=data['dtype'],
                       name=name)


def read_from_csv_TemporalDataFrame(file: Path, sep: str = ',',
                                    time_list: Optional[Union[Collection, 'DType', Literal['*']]] = None,
                                    time_col: Optional[str] = None,
                                    time_points: Optional[Collection[str]] = None) -> 'TemporalDataFrame':
    """
    Read a .csv file into a TemporalDataFrame.

    :param file: a path to the .csv file to read.
    :param sep: delimiter to use for reading the .csv file.
    :param time_list: time points for the dataframe's rows. (see TemporalDataFrame's documentation for more details.)
    :param time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
        given. This column will be used as the time data.
    :param time_points: a list of time points that should exist. This is useful when using the '*' character to
        specify the list of time points that the TemporalDataFrame should cover.

    :return: a TemporalDataFrame built from the .csv file.
    """
    df = pd.read_csv(file, index_col=0, sep=sep)

    if time_col is None:
        time_col = 'Time_Point'

    if time_list is None and time_col == 'Time_Point':
        time_list = df['Time_Point'].values.tolist()
        del df[time_col]
        time_col = None

    return TemporalDataFrame(df, time_list=time_list, time_col_name=time_col, time_points=time_points)


# GPU output --------------------------------------------------------------------------------------
def read_from_dict(data: Dict[str, Dict[Union['DType', str], Union[np.ndarray, pd.DataFrame]]],
                   obs: Optional[Union[pd.DataFrame, 'TemporalDataFrame']] = None,
                   var: Optional[pd.DataFrame] = None,
                   time_points: Optional[pd.DataFrame] = None,
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
    :param time_points: a pandas DataFrame describing the time points.
    :param dtype: the data type for the matrices in VData.
    :param name: an optional name for the loaded VData object.

    :return: a VData object containing the simulation's data
    """
    _data = {}
    _time_points: List[TimePoint] = []
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
                    if matrix_TP not in _time_points:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _time_points.append(matrix_TP)

            check_tp = True

            _layer_data = np.vstack(list(TP_matrices.values()))
            # check data matches columns shape
            if _columns is not None and _layer_data.shape[1] != len(_columns):
                raise ShapeError(f"Layer '{data_type}' has {_layer_data.shape[1]} columns , should have "
                                 f"{len(_columns)}.")

            _loaded_data = pd.DataFrame(np.vstack(list(TP_matrices.values())), columns=_columns)

            _time_list = [_time_points[matrix_index] for matrix_index, matrix in enumerate(TP_matrices.values())
                          for _ in range(len(matrix))]
            generalLogger.debug(f"Computed time list to be : {repr_array(_time_list)}.")

            _data[data_type] = TemporalDataFrame(data=_loaded_data,
                                                 time_points=_time_points,
                                                 time_list=_time_list,
                                                 index=_index,
                                                 dtype=dtype)

            generalLogger.info(f"Loaded layer '{data_type}' ({data_index+1}/{len(data)})")

        # if time points is not given, build a DataFrame from time points found in 'data'
        if time_points is None:
            time_points = pd.DataFrame({"value": _time_points})

        return vdata.VData(_data, obs=obs, var=var, time_points=time_points, dtype=dtype, name=name)


# HDF5 file format --------------------------------------------------------------------------------
def read(file: Union[Path, str], mode: Literal['r', 'r+'] = 'r',
         dtype: Optional['DType'] = None,
         name: Optional[Any] = None,
         backup: bool = False) -> 'vdata.VData':
    """
    Function for reading data from an h5 file and building a VData object from it.

    :param file: path to an h5 file.
    :param mode: reading mode : 'r' (read only) or 'r+' (read and write).
    :param dtype: data type to force on the newly built VData object. If set to None, the dtype is inferred from
        the h5 file.
    :param name: an optional name for the loaded VData object.
    :param backup: create a backup copy of the read h5 file in case something goes wrong ?
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

    data = {'obs': None, 'var': None, 'time_points': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None,
            'uns': {}}

    # import data from file
    importFile = H5GroupReader(File(str(file), mode))
    for key in importFile.keys():
        generalLogger.info(f"Got key : '{key}'.")

        if key in ('obs', 'var', 'time_points', 'layers', 'obsm', 'obsp', 'varm', 'varp', 'uns'):
            type_ = importFile[key].attrs('type')
            data[key] = func_[type_](importFile[key])

        else:
            generalLogger.warning(f"Unexpected data with key {key} while reading file, skipping.")

    new_VData = vdata.VData(data['layers'],
                            data['obs'], data['obsm'], None,
                            data['var'], data['varm'], data['varp'],
                            data['time_points'], data['uns'], dtype=dtype,
                            name=name, file=importFile)

    if data['obsp'] is not None:
        for key, arr in data['obsp'].items():
            new_VData.obsp[key] = arr

    generalLogger.debug("\u23BF read VData : end -------------------------------------------------------- ")

    return new_VData


def read_TemporalDataFrame(file: Union[Path, str], mode: Literal['r', 'r+'] = 'r',
                           dtype: Optional['DType'] = None,
                           name: Optional[Any] = None) -> 'TemporalDataFrame':
    """
    Function for reading data from an h5 file and building a TemporalDataFrame object from it.

    :param file: path to an h5 file.
    :param mode: reading mode : 'r' (read only) or 'r+' (read and write).
    :param dtype: data type to force on the newly built TemporalDataFrame object. If set to None, the dtype is inferred
        from the h5 file.
    :param name: an optional name for the loaded TemporalDataFrame object.

    :return: a loaded TemporalDataFrame.
    """
    file = parse_path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    # import data from file
    import_file = H5GroupReader(File(str(file), mode))
    import_data = import_file[list(import_file.keys())[0]]

    dataset_type = import_data.attrs('type')
    tdf = func_[dataset_type](import_data)

    if name is not None:
        tdf.name = name

    tdf.astype(dtype)

    return tdf


def read_h5_dict(group: H5GroupReader, level: int = 1) -> Dict:
    """
    Function for reading a dictionary from an h5 file.

    :param group: a H5GroupReader from which to read a dictionary.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading dict {group.name}.")

    data = {}

    for dataset_key in group.keys():
        dataset_type = cast(H5GroupReader, group[dataset_key]).attrs("type")

        data[get_value(dataset_key)] = func_[dataset_type](group[dataset_key], level=level+1)

    return data


def read_h5_VDataFrame(group: H5GroupReader, level: int = 1) -> VDataFrame:
    """
    Function for reading a pandas DataFrame from an h5 file.

    :param group: a H5GroupReader from which to read a DataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading VDataFrame {group.name}.")

    # get index
    dataset_type = cast(H5GroupReader, group['index']).attrs("type")
    index = func_[dataset_type](group['index'], level=level + 1)

    # get columns in right order
    data = {}
    for col in group['data'].keys():
        dataset_type = cast(H5GroupReader, group['data'][col]).attrs("type")
        data[get_value(col)] = func_[dataset_type](group['data'][col], level=level + 1)

    if data == {}:
        return VDataFrame(index=index, file=group.group)

    else:
        vdf = VDataFrame(data, file=group.group)
        vdf.index = index

        return vdf


def read_h5_TemporalDataFrame(group: H5GroupReader, level: int = 1) -> 'TemporalDataFrame':
    """
    Function for reading a TemporalDataFrame from an h5 file.

    :param group: a H5GroupReader from which to read a TemporalDataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading TemporalDataFrame {group.name}.")

    # get index
    dataset_type = cast(H5GroupReader, group['index']).attrs("type")
    index = func_[dataset_type](group['index'], level=level + 1)

    # get time_col
    dataset_type = cast(H5GroupReader, group['time_col_name']).attrs("type")
    time_col_name = func_[dataset_type](group['time_col_name'], level=level + 1)

    # get time_list
    dataset_type = cast(H5GroupReader, group['time_list']).attrs("type")
    time_list = func_[dataset_type](group['time_list'], level=level + 1)

    data = {}
    for col in group['data'].keys():
        dataset_type = cast(H5GroupReader, group['data'][col]).attrs("type")
        data[col] = func_[dataset_type](group['data'][col], level=level + 1)

    return TemporalDataFrame(data, time_col_name=time_col_name,
                             index=index, time_list=time_list, name=group.name.split("/")[-1],
                             file=group.group)


def read_h5_chunked_TemporalDataFrame(group: H5GroupReader, level: int = 1) -> 'TemporalDataFrame':
    """
    Function for reading a TemporalDataFrame from an h5 file as DataSets.

    :param group: a H5GroupReader from which to read a TemporalDataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading chunked TemporalDataFrame {group.name}.")

    # get column order
    dataset_type = cast(H5GroupReader, group['columns']).attrs("type")
    columns = func_[dataset_type](group['columns'], level=level + 1)

    # get index
    dataset_type = cast(H5GroupReader, group['index']).attrs("type")
    index = func_[dataset_type](group['index'], level=level + 1)

    # get time_col
    dataset_type = cast(H5GroupReader, group['time_col_name']).attrs("type")
    time_col_name = func_[dataset_type](group['time_col_name'], level=level + 1)

    return TemporalDataFrame(group.group, time_col_name=time_col_name,
                             index=index, columns=columns, name=group.name.split("/")[-1])


def read_h5_series(group: H5GroupReader, index: Optional[List] = None, level: int = 1,
                   log_func: Literal['debug', 'info'] = 'info') -> pd.Series:
    """
    Function for reading a pandas Series from an h5 file.

    :param group: an H5GroupReader from which to read a Series.
    :param index: an optional list representing the indexes for the Series.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    :param log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(level)}Reading Series {group.name}.")

    # simple Series
    if group.isinstance(Dataset):
        data_type = get_dtype_from_string(group.attrs('dtype'))
        values = list(map(data_type, read_h5_array(group, level=level+1, log_func=log_func)))
        return pd.Series(values, index=index)

    # categorical Series
    elif group.isinstance(Group):
        # get data
        data_type = get_dtype_from_string(group.attrs('dtype'))
        categories = read_h5_array(cast(H5GroupReader, group['categories']), level=level+1, log_func=log_func)
        ordered = get_value(group.attrs('ordered'))
        values = list(map(data_type, read_h5_array(cast(H5GroupReader, group['values']), level=level+1,
                                                   log_func=log_func)))

        return pd.Series(pd.Categorical(values, categories, ordered=ordered), index=index)

    # unexpected type
    else:
        raise VTypeError(f"Unexpected type {type(group)} while reading h5 file.")


def read_h5_array(group: H5GroupReader, level: int = 1,
                  log_func: Literal['debug', 'info'] = 'info') -> np.ndarray:
    """
    Function for reading a numpy array from an h5 file.
    If the imported array contains strings, as they where stored as bytes, they are converted back to strings.

    :param group: a H5GroupReader from which to read an array.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    :param log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(level)}Reading array {group.name}.")

    arr = group[()]

    if isinstance(arr, np.ndarray):
        # fix string arrays (object type to strings)
        if arr.dtype.type is np.object_:
            try:
                return arr.astype(float)

            except ValueError:
                return arr.astype(np.str_)

        return arr

    else:
        print(group.name)
        print(arr)
        raise VTypeError(f"Group is not an array (type is '{type(arr)}').")


def read_h5_value(group: H5GroupReader, level: int = 1) -> Union[str, int, float, bool, type]:
    """
    Function for reading a value from an h5 file.

    :param group: a H5GroupReader from which to read a value.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading value {group.name}.")

    if group.isstring():
        return group.asstring()

    return get_value(group[()])


def read_h5_path(group: H5GroupReader, level: int = 1) -> Path:
    """
    Function for reading a Path from an h5 file.

    :param group: a H5GroupReader from which to read a Path.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading Path {group.name}.")

    return Path(group.asstring())


def read_h5_None(_: H5GroupReader, level: int = 1) -> None:
    """
    Function for reading 'None' from an h5 file.

    :param _: a H5GroupReader from which to read a value.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading None.")
    return None


func_: Dict[str, Callable] = {
    'dict': read_h5_dict,
    'VDF': read_h5_VDataFrame,
    'TDF': read_h5_TemporalDataFrame,
    'CHUNKED_TDF': read_h5_chunked_TemporalDataFrame,
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
