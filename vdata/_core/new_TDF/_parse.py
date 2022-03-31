# coding: utf-8
# Created on 29/03/2022 11:43
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from warnings import warn
from numbers import Number
from h5py import File, Group

from typing import Union, Collection, Optional

from vdata import TimePoint
from .name_utils import H5Data


# ====================================================
# code
def parse_data(data: Union[None, dict, pd.DataFrame, H5Data],
               index: Optional[Collection],
               columns_numerical: Optional[Collection],
               columns_string: Optional[Collection],
               time_list: Optional[Collection[Union[Number, str, TimePoint]]],
               time_col_name: Optional[str],
               lock: Optional[tuple[bool, bool]],
               name: str) \
        -> Union[tuple[None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool],
                       Optional[str], str],
                 tuple[H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, tuple[bool, bool], Optional[str], str]]:
    """
    Parse the user-given data to create a TemporalDataFrame.

    Args:
        data: Optional object containing the data to store in this TemporalDataFrame. It can be :
            - a dictionary of ['column_name': [values]], where [values] has always the same length
            - a pandas DataFrame
            - a H5 File or Group containing numerical and string data.
        index: Optional indices to set or to substitute to indices defined in data if data is a pandas DataFrame.
        columns_numerical: Optional numerical column names to set or to substitute to numerical columns defined in
            data if data is a pandas DataFrame.
        columns_string: Optional string column names to set or to substitute to string columns defined in data if data
            is a pandas DataFrame.
        time_list: Optional list of time values of the same length as the index, indicating for each row at which
            time point it exists.
        time_col_name: Optional column name in data (if data is a dict or a pandas DataFrame) to use as time data.
        lock: Optional 2-tuple of booleans indicating which axes (index, columns) are locked.
        name: A name for the TemporalDataFrame.

    Returns:
        A H5 file (when data is backed on a file),
        the numerical and string arrays of data,
        the indices,
        the numerical column names, the string column names
        the lock
        the time_col_name
        and the name.
    """
    if data is None:
        if time_list is not None:
            warn("No data supplied, 'time_list' parameter is ignored.")

        if time_col_name is not None:
            warn("No data supplied, 'time_col_name' parameter is ignored.")

        return None, \
            np.array([]), np.array([]), np.array([]), \
            np.array([]) if index is None else index, \
            np.array([]) if columns_numerical is None else columns_numerical, \
            np.array([]) if columns_string is None else columns_string,\
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), None, str(name)

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        numerical_array, string_array, timepoints_array, index, columns_numerical, columns_string, time_col_name = \
            parse_data_df(data, index, columns_numerical, columns_string, time_list, time_col_name)

        return None, numerical_array, string_array, timepoints_array, index, columns_numerical, columns_string, \
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), time_col_name, str(name)

    if isinstance(data, (File, Group)):
        if time_list is not None:
            warn("Data loaded from a H5 file, 'time_list' parameter is ignored.")

        if time_col_name is not None:
            warn("Data loaded from a H5 file, 'time_col_name' parameter is ignored.")

        return parse_data_h5(data, index, columns_numerical, columns_string, lock, name)

    raise ValueError("Invalid 'data' supplied for creating a TemporalDataFrame. 'data' can be :\n"
                     " - a dictionary of ['column_name': [values]], where [values] has always the same length \n"
                     " - a pandas DataFrame \n"
                     " - a H5 File or Group containing numerical and string data")


def parse_data_df(data: pd.DataFrame,
                  index: Optional[Collection],
                  columns_numerical: Optional[Collection],
                  columns_string: Optional[Collection],
                  time_list: Optional[Collection[Union[Number, str, TimePoint]]],
                  time_col_name: Optional[str]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
    numerical_df = data.select_dtypes(include='number')
    string_df = data.select_dtypes(exclude='number')

    if index is not None:
        if (l := len(index)) != (s := data.shape[0]):
            raise ValueError(f"'index' parameter has incorrect length {l}, expected {s}.")

    else:
        index = data.index.values

    if columns_numerical is not None:
        if (l := len(columns_numerical)) != (s := numerical_df.shape[1]):
            raise ValueError(f"'columns_numerical' parameter has incorrect length {l}, expected {s}.")

    else:
        columns_numerical = numerical_df.columns.values

    if columns_string is not None:
        if (l := len(columns_string)) != (s := string_df.shape[1]):
            raise ValueError(f"'columns_string' parameter has incorrect length {l}, expected {s}.")

    else:
        columns_string = string_df.columns.values

    if time_list is None:
        if time_col_name is None:
            # set time values to default value '0'
            time_values = [TimePoint(0) for _ in enumerate(index)]

        else:
            # pick time values from df
            if time_col_name not in data.columns:
                raise ValueError(f"'time_col_name' ('{time_col_name}') is not in the data's columns.")

            if time_col_name in numerical_df.columns:
                time_values = list(map(TimePoint, numerical_df[time_col_name].values))
                del numerical_df[time_col_name]

            else:
                time_values = list(map(TimePoint, string_df[time_col_name].values))
                del string_df[time_col_name]

    else:
        # pick time values from time_list
        if time_col_name is not None:
            warn("'time_list' parameter already supplied, 'time_col_name' parameter is ignored.")

        if (l := len(time_list)) != (li := len(index)):
            raise ValueError(f"'time_list' parameter has incorrect length {l}, expected {li}.")

        time_values = list(map(TimePoint, time_list))

    numerical_array = numerical_df.values.copy()
    if numerical_df.empty:
        numerical_array = numerical_array.astype(int)

    string_array = string_df.values.copy()
    if string_df.empty:
        string_array = string_array.astype('O')

    return numerical_array, string_array, np.array(time_values), np.array(index), \
        np.array(columns_numerical), np.array(columns_string), time_col_name


def parse_data_h5(data: H5Data,
                  index: Optional[Collection],
                  columns_numerical: Optional[Collection],
                  columns_string: Optional[Collection],
                  lock: Optional[tuple[bool, bool]],
                  name: str) \
        -> tuple[H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, tuple[bool, bool], Optional[str], str]:
    numerical_array_file = data['values_numerical']
    string_array_file = data['values_string']
    timepoints_array_file = data['timepoints']
    index_file = data['index']
    columns_num_file = data['columns_numerical']
    columns_str_file = data['columns_string']

    if index is not None:
        if len(index) != len(index_file):
            raise ValueError(f"'index' parameter has incorrect length {len(index)}, expected {len(index_file)}.")

        index_file[()] = index

    if columns_numerical is not None:
        if len(columns_numerical) != len(columns_num_file):
            raise ValueError(f"'columns_numerical' parameter has incorrect length {len(columns_numerical)}, "
                             f"expected {len(columns_num_file)}.")

        columns_num_file[()] = columns_numerical

    if columns_string is not None:
        if len(columns_string) != len(columns_str_file):
            raise ValueError(f"'columns_string' parameter has incorrect length {len(columns_string)}, "
                             f"expected {len(columns_str_file)}.")

        columns_str_file[()] = columns_string

    if lock is not None:
        data.attrs['locked_indices'], data.attrs['locked_columns'] = bool(lock[0]), bool(lock[1])

    timepoints_columns_name = None if data.attrs['time_points_column_name'] == '__TDF_None__' else data.attrs[
        'time_points_column_name']

    if name != 'No_Name':
        data.attrs['name'] = str(name)

    return data, numerical_array_file, string_array_file, timepoints_array_file, index_file, columns_num_file, \
        columns_str_file, (data.attrs['locked_indices'], data.attrs['locked_columns']), timepoints_columns_name, \
        data.attrs['name']
