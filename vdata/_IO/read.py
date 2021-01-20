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
from typing import Union, Optional, Dict, List, AbstractSet, ValuesView, Any, cast, Callable, Tuple, Collection
from typing_extensions import Literal

import vdata
from .utils import parse_path
from .logger import generalLogger, getLoggingLevel
from .errors import VValueError, VTypeError
from ..NameUtils import DType, ArrayLike_2D, H5Group
from .._core import utils


def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


# ====================================================
# code
# CSV file format ---------------------------------------------------------------------------------
def read_from_csv(directory: Union[Path, str], dtype: DType = np.float32,
                  time_list: Optional[Union[Collection, DType, Literal['*']]] = None,
                  time_col: Optional[str] = None,
                  time_points: Optional[Collection[str]] = None) -> 'vdata.VData':
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
            'varm': None, 'varp': None}

    # import the data
    for f in os.listdir(directory):
        if f != ".metadata.json":
            generalLogger.info(f"Got key : '{f}'.")

            if f.endswith('.csv'):
                if f in ('var.csv', 'time_points.csv'):
                    generalLogger.info(f"{spacer(1)}Reading pandas DataFrame '{f[:-4]}'.")
                    data[f[:-4]] = pd.read_csv(directory / f, index_col=0)

                else:
                    generalLogger.info(f"{spacer(1)}Reading TemporalDataFrame '{f[:-4]}'.")
                    if time_list is None and time_col is None:
                        if metadata['obs']['time_points_column_name'] != '__TPID':
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
                       data['time_points'], dtype=dtype)


def TemporalDataFrame_read_csv(file: Path, sep: str = ',',
                               time_list: Optional[Union[Collection, DType, Literal['*']]] = None,
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

def read_from_dict(data: Dict[str, Dict[Union[DType, str], ArrayLike_2D]], obs: Optional[pd.DataFrame] = None,
                   var: Optional[pd.DataFrame] = None, time_points: Optional[pd.DataFrame] = None,
                   dtype: DType = np.float32) -> 'vdata.VData':
    """
    Load a simulation's recorded information into a VData object.

    If time points are not given explicitly, this function will try to recover them from the time point names in
        the data.
    For this to work, time points names must be strings with :
        - last character in (s, m, h, D, M, Y)
        - first characters convertible to a float
    The last character indicates the unit:
        - s : second
        - m : minute
        - h : hour
        - D : day
        - M : month
        - Y : year

    :param data: a dictionary of data types (RNA, Proteins, etc.) linked to dictionaries of time points linked to
        matrices of cells x genes
    :param obs: a pandas DataFrame describing the observations (cells)
    :param var: a pandas DataFrame describing the variables (genes)
    :param time_points: a pandas DataFrame describing the time points
    :param dtype: the data type for the matrices in VData

    :return: a VData object containing the simulation's data
    """
    time_point_units = {
        's': 'second',
        'm': 'minute',
        'h': 'hour',
        'D': 'day',
        'M': 'month',
        'Y': 'year',
    }

    _data = {}
    _time_points = []
    check_tp = False

    if not isinstance(data, dict):
        raise VTypeError("Data should be a dictionary with format : {data type: {time point: matrix}}")

    else:
        for data_type, TP_matrices in data.items():
            if not isinstance(TP_matrices, dict):
                raise VTypeError(f"'{data_type}' in data should be a dictionary with format : {{time point: matrix}}")

            for matrix_index, matrix in TP_matrices.items():
                if not isinstance(matrix, (np.ndarray, pd.DataFrame)) or matrix.ndim != 2:
                    raise VTypeError(f"Item at time point '{matrix_index}' is not a 2D array-like object "
                                     f"(numpy ndarray, pandas DatFrame).")

                elif check_tp:
                    if matrix_index not in _time_points:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _time_points.append(matrix_index)

            check_tp = True

            _data[data_type] = np.array([np.array(matrix) for matrix in TP_matrices.values()])

        # if time points not given, try to guess them
        if time_points is None:
            if all([isinstance(_time_points[i], str) for i in range(len(_time_points))]):
                TP_data: Dict[str, List[Union[float, str]]] = {"value": [], "unit": []}
                del_unit = False

                for tp in _time_points:
                    tp = cast(str, tp)          # for typing

                    if tp[-1] in time_point_units.keys():
                        try:
                            TP_data["value"].append(float(tp[:-1]))
                            TP_data["unit"].append(time_point_units[tp[-1]])

                        except ValueError:
                            del_unit = True
                            break

                    else:
                        del_unit = True
                        break

                if del_unit:
                    TP_data = {"value": _time_points}

                TP_df = pd.DataFrame(TP_data)

            else:
                TP_df = pd.DataFrame({"value": _time_points})

            return vdata.VData(_data, obs=obs, var=var, time_points=TP_df, dtype=dtype)

        else:
            return vdata.VData(_data, obs=obs, var=var, time_points=time_points, dtype=dtype)


# HDF5 file format --------------------------------------------------------------------------------
class H5GroupReader:
    """
    Class for reading a h5py File, Group or Dataset
    """

    def __init__(self, group: H5Group):
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
        elif key is ():
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


def read(file: Union[Path, str], dtype: Optional[DType] = None) -> 'vdata.VData':
    """
    Function for reading data from a .h5 file and building a VData object from it.

    :param file: path to a .h5 file.
    :param dtype: data type to force on the newly built VData object. If set to None, the dtype is inferred from
        the .h5 file.
    """
    file = parse_path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    data = {'obs': None, 'var': None, 'time_points': None,
            'layers': None,
            'obsm': None, 'obsp': None,
            'varm': None, 'varp': None}

    uns: Optional[Dict] = None

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
                       data['time_points'], uns, dtype)


def read_h5_dict(group: H5GroupReader, level: int = 1) -> Dict:
    """
    Function for reading a dictionary from a .h5 file.

    :param group: a H5GroupReader from which to read a dictionary.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading dict {group.name}.")

    data = {}

    for dataset_key in group.keys():
        dataset_type = group[dataset_key].attrs("type")

        data[dataset_key] = func_[dataset_type](group[dataset_key], level=level+1)

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
        data[col] = read_h5_series(group[col], index, level=level+1)

    return pd.DataFrame(data, index=index)


def read_h5_TemporalDataFrame(group: H5GroupReader, level: int = 1) -> 'vdata.TemporalDataFrame':
    """
    Function for reading a TemporalDataFrame from a .h5 file.

    :param group: a H5GroupReader from which to read a TemporalDataFrame.
    :param level: for logging purposes, the recursion depth of calls to a read_h5 function.
    """
    generalLogger.info(f"{spacer(level)}Reading TemporalDataFrame {group.name}.")

    # get column order
    col_order = group.attrs('column_order')
    # get index
    index = group.attrs('index')

    # get columns in right order
    data = {}
    time_list = None

    log_func: Literal['debug', 'info'] = 'info'

    for i, col in enumerate(col_order):
        if col == '__TPID':
            time_list = read_h5_series(group[col], index, level=level+1)

        else:
            data[col] = read_h5_series(group[col], index, level=level+1, log_func=log_func)

        if log_func == 'info' and i > 0:
            log_func = 'debug'

    if getLoggingLevel() != 'DEBUG':
        generalLogger.info(f"{spacer(level+1)}...")

    return vdata.TemporalDataFrame(data, time_list=time_list, index=index, name=group.name.split("/")[-1])


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
        ordered = group.attrs('ordered')
        values = read_h5_array(group['values'], level=level+1, log_func=log_func)

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


func_: Dict[str, Callable] = {
    'dict': read_h5_dict,
    'DF': read_h5_DataFrame,
    'TDF': read_h5_TemporalDataFrame,
    'series': read_h5_series,
    'array': read_h5_array,
    'value': read_h5_value,
    'type': read_h5_value
}
