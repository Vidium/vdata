# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
import os
import h5py
import pickle
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Union, Optional, Dict, List, AbstractSet, ValuesView, Any, cast

from .logger import generalLogger
from .errors import VValueError, VTypeError
from ..NameUtils import DType, DTypes, ArrayLike_2D, ArrayLike, LoggingLevel, H5Group
from .._core.vdata import VData

spacer = "  " + u'\u21B3' + " "


# ====================================================
# code
# TODO : deprecated, remove
def read_pickle(file: Union[Path, str]) -> VData:
    """
    Load a pickled VData object.
    Example :
    '>>> import vdata
    '>>> vdata.read("/path/to/file.p")

    :param file: path to a saved VData object
    """
    # make sure file is a path
    if not isinstance(file, Path):
        file = Path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    with open(file, 'rb') as save_file:
        vdata = pickle.load(save_file)

    return vdata


# CSV file format ---------------------------------------------------------------------------------
# TODO
def read_from_csv(directory: Union[Path, str]) -> VData:
    """
    TODO
    """
    pass


# GPU output --------------------------------------------------------------------------------------

def read_from_GPU(data: Dict[str, Dict[Union[DType, str], ArrayLike_2D]], obs: Optional[pd.DataFrame] = None, var: Optional[pd.DataFrame] = None, time_points: Optional[pd.DataFrame] = None,
                  dtype: DType = np.float32, log_level: LoggingLevel = "INFO") -> VData:
    """
    Load a simulation's recorded information into a VData object.

    If time points are not given explicitly, this function will try to recover them from the time point names in the data.
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

    :param data: a dictionary of data types (RNA, Proteins, etc.) linked to dictionaries of time points linked to matrices of cells x genes
    :param obs: a pandas DataFrame describing the observations (cells)
    :param var: a pandas DataFrame describing the variables (genes)
    :param time_points: a pandas DataFrame describing the time points
    :param dtype: the data type for the matrices in VData
    :param log_level: the logging level for the VData, in (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    :return: a VData object containing the simulation's data
    """
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
                if not isinstance(matrix, (np.ndarray, sparse.spmatrix, pd.DataFrame)) or matrix.ndim != 2:
                    raise VTypeError(f"Item at time point '{matrix_index}' is not a 2D array-like object (numpy ndarray, scipy sparse matrix, pandas DatFrame).")

                elif check_tp:
                    if matrix_index not in _time_points:
                        raise VValueError("Time points do not match for all data types.")
                else:
                    _time_points.append(matrix_index)

            check_tp = True

            _data[data_type] = np.array([matrix for matrix in TP_matrices.values()])

        # if time points not given, try to guess them
        if time_points is None:
            if all([isinstance(_time_points[i], str) for i in range(len(_time_points))]):
                TP_df_data: Dict[str, List[Union[float, str]]] = {"value": [], "unit": []}
                del_unit = False

                for tp in _time_points:
                    tp = cast(str, tp)          # for typing
                    if tp.endswith("s"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("second")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("m"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("minute")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("h"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("hour")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("D"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("day")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("M"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("month")
                        except ValueError:
                            del_unit = True
                            break
                    elif tp.endswith("Y"):
                        try:
                            TP_df_data["value"].append(float(tp[:-1]))
                            TP_df_data["unit"].append("year")
                        except ValueError:
                            del_unit = True
                            break
                    else:
                        del_unit = True
                        break

                if del_unit:
                    TP_df_data = {"value": _time_points}

                TP_df = pd.DataFrame(TP_df_data)

            else:
                TP_df = pd.DataFrame({"value": _time_points})

            return VData(_data, obs=obs, var=var, time_points=TP_df, dtype=dtype, log_level=log_level)

        else:
            return VData(_data, obs=obs, var=var, time_points=time_points, dtype=dtype, log_level=log_level)


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

    def __getitem__(self, key: Union[str, slice]) -> Union['H5GroupReader', np.ndarray]:
        """
        Get a sub-group from the group, identified by a key

        :param key: the name of the sub-group
        """
        if isinstance(key, slice):
            return self._check_type(self.group[:])
        elif key is ...:
            return self._check_type(self.group[...])
        else:
            return H5GroupReader(self.group[key])

    def __enter__(self):
        self.group.__enter__()
        return self

    def __exit__(self, *_):
        self.group.__exit__()

    def keys(self) -> AbstractSet:
        """
        Get keys of the group
        """
        return self.group.keys()

    def values(self) -> ValuesView:
        """
        Get values of the group
        """
        return self.group.values()

    def items(self) -> AbstractSet:
        """
        Get (key, value) tuples of the group
        """
        return self.group.items()

    def attr(self, key: str) -> Any:
        """
        Get an attribute, identified by a key, from the group

        :param key: the name of the attribute
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


def read(file: Union[Path, str], dtype: Optional[DType] = None, log_level: Optional[LoggingLevel] = None) -> VData:
    """
    Function for reading data from a .h5 file and building a VData object from it.

    :param file: path to a .h5 file.
    :param dtype: data type to force on the newly built VData object. If set to None, the dtype is inferred from the .h5 file.
    :param log_level: logger level to force on the newly built VData object. If set to None, the level is inferred from the .h5 file.
    """
    if log_level is not None:
        generalLogger.set_level(log_level)

    # make sure file is a path
    if not isinstance(file, Path):
        file = Path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    data: Dict[str, Optional[Union[ArrayLike, Dict]]] = {'layers': None,
                                                         'obs': None, 'obsm': None, 'obsp': None,
                                                         'var': None, 'varm': None, 'varp': None,
                                                         'time_points': None,
                                                         'uns': None}

    # import data from file
    with H5GroupReader(h5py.File(file, "r")) as importFile:
        for key in importFile.keys():
            generalLogger.info(f"reading {key}")

            if key in ("obs", "var", "time_points"):
                data[key] = read_h5_dataframe(importFile[key])
                generalLogger.debug(spacer + "\n" + str(data[key]))

            elif key in ('layers', 'obsm', 'obsp', 'varm', 'varp'):
                data[key] = {}
                for dataset_key in importFile[key].keys():
                    data[key][dataset_key] = read_h5_array(importFile[key][dataset_key])
                generalLogger.debug(spacer + "\n" + str(data[key]))

            elif key == 'uns':
                data['uns'] = read_h5_dict(importFile[key])
                generalLogger.debug(spacer + "\n" + str(data['uns']))

            elif key == 'dtype':
                dtype = DTypes[str(importFile[key][...])]
                generalLogger.debug(spacer + str(dtype))

            elif key == 'log_level':
                log_level = str(importFile[key][...])
                generalLogger.debug(spacer + log_level)

            else:
                generalLogger.warning(f"Unexpected data with key {key} while reading file, skipping.")

    return VData(data['layers'], data['obs'], data['obsm'], data['obsp'], data['var'], data['varm'], data['varp'], data['time_points'], data['uns'], dtype, log_level)


def read_h5_dict(group: H5GroupReader) -> Dict:
    """
    Function for reading a dictionary from a .h5 file.

    :param group: a H5GroupReader from which to read a dictionary
    """
    func_ = {'dict': read_h5_dict,
             'DF': read_h5_dataframe,
             'series': read_h5_series,
             'array': read_h5_array,
             'value': read_h5_value,
             'type': read_h5_value}

    data = {}

    for dataset_key in group.keys():
        dataset_type = group[dataset_key].attr("type")

        data[dataset_key] = func_[dataset_type](group[dataset_key])

    return data


def read_h5_dataframe(group: H5GroupReader) -> pd.DataFrame:
    """
    Function for reading a pandas DataFrame from a .h5 file.

    :param group: a H5GroupReader from which to read a DataFrame
    """
    # get column order
    col_order = group.attr('column_order')
    # get index
    index = group.attr('index')

    # get columns in right order
    df_data = {}
    for col in col_order:
        df_data[col] = read_h5_series(group[col], index)

    return pd.DataFrame(df_data, index=index)


def read_h5_series(group: H5GroupReader, index: Optional[List] = None) -> pd.Series:
    """
    Function for reading a pandas Series from a .h5 file.

    :param group: an H5GroupReader from which to read a Series
    :param index: an optional list representing the indexes for the Series
    """
    # simple Series
    if group.isinstance(h5py.Dataset):
        return pd.Series(read_h5_array(group), index=index)

    # categorical Series
    elif group.isinstance(h5py.Group):
        # get data
        categories = group.attr('categories')
        ordered = group.attr('ordered')
        values = read_h5_array(group['values'])

        return pd.Series(pd.Categorical(values, categories, ordered=ordered), index=index)

    # unexpected type
    else:
        raise VTypeError(f"Unexpected type {type(group)} while reading .h5 file.")


def read_h5_array(group: H5GroupReader) -> np.ndarray:
    """
    Function for reading a numpy array from a .h5 file.
    If the imported array contains strings, as they where stored as bytes, they are converted back to strings.

    :param group: a H5GroupReader from which to read an array
    """
    arr = group[:]

    # fix string arrays (object type to strings)
    if arr.dtype.type is np.object_:
        return arr.astype(np.str_)

    return arr


def read_h5_value(group: H5GroupReader) -> Union[str, int, float, bool, type]:
    """
    Function for reading a numpy array from a .h5 file.
    If the imported array contains strings, as they where stored as bytes, they are converted back to strings.

    :param group: a H5GroupReader from which to read an array
    """
    print(group[...], type(group[...]))
    return cast(Union[str, int, float, bool, type], group[...])