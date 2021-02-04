# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
import os
import h5py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, AbstractSet, ValuesView, Any, Callable, Tuple, Collection, cast
from typing_extensions import Literal

import vdata
from .utils import parse_path
from .logger import generalLogger, getLoggingLevel
from .errors import VValueError, VTypeError
from .. import utils
from .. import NameUtils


def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


# ====================================================
# code
# CSV file format ---------------------------------------------------------------------------------
def read_from_csv(directory: Union[Path, str],
                  dtype: 'NameUtils.DType' = np.float32,
                  time_list: Optional[Union[Collection, 'NameUtils.DType', Literal['*']]] = None,
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
                            time_col = metadata['obs']['time_points_column_name']

                    data[f[:-4]] = TemporalDataFrame_read_csv(directory / f, time_list=time_list,
                                                              time_col=time_col, time_points=time_points)

            else:
                dataset_dict = {}
                generalLogger.info(f"{spacer(1)}Reading group '{f}'.")

                for dataset in os.listdir(directory / f):
                    if f in ('layers', ):
                        generalLogger.info(f"{spacer(2)} Reading {dataset}")
                        dataset_dict[dataset[:-4]] = TemporalDataFrame_read_csv(directory / f / dataset,
                                                                                time_list=time_list,
                                                                                time_points=time_points)

                    else:
                        raise NotImplementedError

                data[f] = dataset_dict

    return vdata.VData(data['layers'],
                       data['obs'], data['obsm'], data['obsp'],
                       data['var'], data['varm'], data['varp'],
                       data['time_points'], dtype=data['dtype'],
                       name=name)


def TemporalDataFrame_read_csv(file: Path, sep: str = ',',
                               time_list: Optional[Union[Collection, 'NameUtils.DType', Literal['*']]] = None,
                               time_col: Optional[str] = None,
                               time_points: Optional[Collection[str]] = None) -> 'vdata.TemporalDataFrame':
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
    df = pd.read_csv(file, index_col=0, sep=sep, )

    if time_list is None and time_col is None:
        time_list = df['Time_Point'].values.tolist()

    del df['Time_Point']

    return vdata.TemporalDataFrame(df, time_list=time_list, time_col=time_col, time_points=time_points)


# GPU output --------------------------------------------------------------------------------------

def read_from_dict(data: Dict[str, Dict[Union['NameUtils.DType', str], 'NameUtils.ArrayLike_2D']],
                   obs: Optional[Union[pd.DataFrame, 'vdata.TemporalDataFrame']] = None,
                   var: Optional[pd.DataFrame] = None,
                   time_points: Optional[pd.DataFrame] = None,
                   dtype: 'NameUtils.DType' = np.float32,
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
    _time_points: List[vdata.TimePoint] = []
    check_tp = False

    if not isinstance(data, dict):
        raise VTypeError("Data should be a dictionary with format : {data type: {time point: matrix}}")

    else:
        for data_index, (data_type, TP_matrices) in enumerate(data.items()):
            if not isinstance(TP_matrices, dict):
                raise VTypeError(f"'{data_type}' in data should be a dictionary with format : {{time point: matrix}}")

            generalLogger.debug(f"Loading layer '{data_type}'.")

            for matrix_index, matrix in TP_matrices.items():
                matrix_TP = vdata.TimePoint(matrix_index)

                if not isinstance(matrix, (np.ndarray, pd.DataFrame)) or matrix.ndim != 2:
                    raise VTypeError(f"Item at time point '{matrix_TP}' is not a 2D array-like object "
                                     f"(numpy ndarray, pandas DatFrame).")

                elif check_tp:
                    if matrix_TP not in _time_points:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _time_points.append(matrix_TP)

            check_tp = True

            index = obs.index if obs is not None else None
            columns = var.index if var is not None else None

            generalLogger.debug(f"Found index is : {utils.repr_array(index)}.")
            generalLogger.debug(f"Found columns is : {utils.repr_array(columns)}.")

            loaded_data = pd.DataFrame(np.vstack(list(TP_matrices.values())))

            time_list = [_time_points[matrix_index] for matrix_index, matrix in enumerate(TP_matrices.values())
                         for _ in range(len(matrix))]
            generalLogger.debug(f"Computed time list to be : {utils.repr_array(time_list)}.")

            _data[data_type] = vdata.TemporalDataFrame(data=loaded_data,
                                                       time_points=_time_points,
                                                       time_list=time_list,
                                                       index=index,
                                                       columns=columns,
                                                       dtype=dtype)

            generalLogger.info(f"Loaded layer '{data_type}' ({data_index+1}/{len(data)})")

        # if time points not given, build a DataFrame from time points found in 'data'
        if time_points is None:
            time_points = pd.DataFrame({"value": _time_points})

        return vdata.VData(_data, obs=obs, var=var, time_points=time_points, dtype=dtype, name=name)


# HDF5 file format --------------------------------------------------------------------------------
class H5GroupReader:
    """
    Class for reading a h5py File, Group or Dataset
    """

    def __init__(self, group: 'NameUtils.H5Group'):
        """
        :param group: a h5py File, Group or Dataset
        """
        self.group = group

    def __getitem__(self, key: Union[str, slice, 'ellipsis', Tuple[()]]) \
            -> Union['H5GroupReader', np.ndarray, str, int, float, bool, type]:
        """
        Get a sub-group from the group, identified by a key

        :param key: the name of the sub-group
        """
        if isinstance(key, slice):
            return self._check_type(self.group[:])
        elif key is ...:
            return self._check_type(self.group[...])
        elif key == ():
            return self._check_type(self.group[()])
        else:
            return H5GroupReader(self.group[key])

    def __enter__(self):
        self.group.__enter__()
        return self

    def __exit__(self, *_):
        self.group.__exit__()

    @property
    def name(self) -> str:
        """
        Get the name of the group.
        :return: the group's name.
        """
        return self.group.name

    def keys(self) -> AbstractSet:
        """
        Get keys of the group.
        :return: the keys of the group.
        """
        return self.group.keys()

    def values(self) -> ValuesView:
        """
        Get values of the group.
        :return: the values of the group.
        """
        return self.group.values()

    def items(self) -> AbstractSet:
        """
        Get (key, value) tuples of the group.
        :return: the items of the group.
        """
        return self.group.items()

    def attrs(self, key: str) -> Any:
        """
        Get an attribute, identified by a key, from the group.

        :param key: the name of the attribute.
        :return: the attribute identified by the key, from the group.
        """
        # get attribute from group
        attribute = self.group.attrs[key]

        return self._check_type(attribute)

    @staticmethod
    def _check_type(data: Any) -> Any:
        """
        Convert data into the expected types.

        :param data: any object which type should be checked.
        """
        # if attribute is an array of bytes, convert bytes to strings
        if isinstance(data, (np.ndarray, np.generic)) and data.dtype.type is np.bytes_:
            return data.astype(np.str_)

        elif isinstance(data, np.ndarray) and data.ndim == 0:
            if data.dtype.type is np.int_:
                return int(data)

            elif data.dtype.type is np.float_:
                return float(data)

            elif data.dtype.type is np.str_ or data.dtype.type is np.object_:
                return str(data)

            elif data.dtype.type is np.bool_:
                return bool(data)

        return data

    def isinstance(self, _type: type) -> bool:
        return isinstance(self.group, _type)


def read(file: Union[Path, str], dtype: Optional['NameUtils.DType'] = None,
         name: Optional[Any] = None) -> 'vdata.VData':
    """
    Function for reading data from a .h5 file and building a VData object from it.

    :param file: path to a .h5 file.
    :param dtype: data type to force on the newly built VData object. If set to None, the dtype is inferred from
        the .h5 file.
    :param name: an optional name for the loaded VData object.
    """
    file = parse_path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    data = {'obs': None, 'var': None, 'time_points': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None,
            'uns': None, 'dtype': dtype}

    # import data from file
    with H5GroupReader(h5py.File(file, "r")) as importFile:
        for key in importFile.keys():
            generalLogger.info(f"Got key : '{key}'.")

            if key in ('obs', 'var', 'time_points', 'layers', 'obsm', 'obsp', 'varm', 'varp', 'uns', 'dtype'):
                type_ = importFile[key].attrs('type')
                data[key] = func_[type_](importFile[key])

            else:
                generalLogger.warning(f"Unexpected data with key {key} while reading file, skipping.")

    return vdata.VData(data['layers'],
                       data['obs'], data['obsm'], data['obsp'],
                       data['var'], data['varm'], data['varp'],
                       data['time_points'], data['uns'], dtype=data['dtype'],
                       name=name)


def read_h5_dict(group: H5GroupReader, level: int = 1) -> Dict:
    """
    Function for reading a dictionary from a .h5 file.

    :param group: a H5GroupReader from which to read a dictionary.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading dict {group.name}.")

    data = {}

    for dataset_key in group.keys():
        dataset_type = cast(H5GroupReader, group[dataset_key]).attrs("type")

        data[utils.get_value(dataset_key)] = func_[dataset_type](group[dataset_key], level=level+1)

    return data


def read_h5_DataFrame(group: H5GroupReader, level: int = 1) -> pd.DataFrame:
    """
    Function for reading a pandas DataFrame from a .h5 file.

    :param group: a H5GroupReader from which to read a DataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading DataFrame {group.name}.")

    # get column order
    col_order = group.attrs('column_order')
    # get index
    index = group.attrs('index')

    # get columns in right order
    data = {}
    for col in col_order:
        data[utils.get_value(col)] = read_h5_series(group[col], index, level=level+1)

    return pd.DataFrame(data, index=index)


def read_h5_TemporalDataFrame(group: H5GroupReader, level: int = 1) -> 'vdata.TemporalDataFrame':
    """
    Function for reading a TemporalDataFrame from a .h5 file.

    :param group: a H5GroupReader from which to read a TemporalDataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading TemporalDataFrame {group.name}.")

    # get column order
    dataset_type = cast(H5GroupReader, group['column_order']).attrs("type")
    col_order = func_[dataset_type](group['column_order'], level=level + 1)

    # get index
    dataset_type = cast(H5GroupReader, group['index']).attrs("type")
    index = func_[dataset_type](group['index'], level=level + 1)

    # get time_col
    dataset_type = cast(H5GroupReader, group['time_col']).attrs("type")
    time_col = func_[dataset_type](group['time_col'], level=level + 1)

    # get time_list
    if time_col is None:
        dataset_type = cast(H5GroupReader, group['time_list']).attrs("type")
        time_list = func_[dataset_type](group['time_list'], level=level + 1)
    else:
        time_list = None

    # get columns in right order
    data = {}

    log_func: Literal['debug', 'info'] = 'info'

    for i, col in enumerate(col_order):
        data[col] = read_h5_series(cast(H5GroupReader, group[str(col)]), index, level=level+1, log_func=log_func)

        if log_func == 'info' and i > 0:
            log_func = 'debug'

    if getLoggingLevel() != 'DEBUG':
        generalLogger.info(f"{spacer(level+1)}...")

    return vdata.TemporalDataFrame(data, time_list=time_list, time_col=time_col,
                                   index=index, name=group.name.split("/")[-1])


def read_h5_series(group: H5GroupReader, index: Optional[List] = None, level: int = 1,
                   log_func: Literal['debug', 'info'] = 'info') -> pd.Series:
    """
    Function for reading a pandas Series from a .h5 file.

    :param group: an H5GroupReader from which to read a Series.
    :param index: an optional list representing the indexes for the Series.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    :param log_func: function to use with the logger. Either 'info' or 'debug'.
    """
    getattr(generalLogger, log_func)(f"{spacer(level)}Reading Series {group.name}.")

    # simple Series
    if group.isinstance(h5py.Dataset):
        return pd.Series(read_h5_array(group, level=level+1, log_func=log_func), index=index)

    # categorical Series
    elif group.isinstance(h5py.Group):
        # get data
        categories = group.attrs('categories')
        ordered = utils.get_value(group.attrs('ordered'))
        values = read_h5_array(cast(H5GroupReader, group['values']), level=level+1, log_func=log_func)

        return pd.Series(pd.Categorical(values, categories, ordered=ordered), index=index)

    # unexpected type
    else:
        raise VTypeError(f"Unexpected type {type(group)} while reading .h5 file.")


def read_h5_array(group: H5GroupReader, level: int = 1,
                  log_func: Literal['debug', 'info'] = 'info') -> np.ndarray:
    """
    Function for reading a numpy array from a .h5 file.
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
            return arr.astype(np.str_)

        return arr

    else:
        raise VTypeError("Group is not an array.")


def read_h5_value(group: H5GroupReader, level: int = 1) -> Union[str, int, float, bool, type]:
    """
    Function for reading a value from a .h5 file.

    :param group: a H5GroupReader from which to read a value.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading value {group.name}.")
    return utils.get_value(group[()])


def read_h5_None(_: H5GroupReader, level: int = 1) -> None:
    generalLogger.info(f"{spacer(level)}Reading None.")
    return None


func_: Dict[str, Callable] = {
    'dict': read_h5_dict,
    'DF': read_h5_DataFrame,
    'TDF': read_h5_TemporalDataFrame,
    'series': read_h5_series,
    'array': read_h5_array,
    'value': read_h5_value,
    'type': read_h5_value,
    'None': read_h5_None
}
