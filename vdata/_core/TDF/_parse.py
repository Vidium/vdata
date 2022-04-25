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

from vdata.utils import isCollection
from vdata.time_point import TimePoint
from .name_utils import H5Data


# ====================================================
# code
def parse_data(data: Union[None, dict, pd.DataFrame, H5Data],
               index: Optional[Collection],
               repeating_index: bool,
               columns_numerical: Optional[Collection],
               columns_string: Optional[Collection],
               time_list: Optional[Collection[Union[Number, str, TimePoint]]],
               time_col_name: Optional[str],
               lock: Optional[tuple[bool, bool]],
               name: str) \
        -> Union[tuple[None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool],
                       Optional[str], str, bool],
                 tuple[H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, tuple[bool, bool], Optional[str],
                       str], bool]:
    """
    Parse the user-given data to create a TemporalDataFrame.

    Args:
        data: Optional object containing the data to store in this TemporalDataFrame. It can be :
            - a dictionary of ['column_name': [values]], where [values] has always the same length
            - a pandas DataFrame
            - a H5 File or Group containing numerical and string data.
        index: Optional indices to set or to substitute to indices defined in data if data is a pandas DataFrame.
        repeating_index: Is the index repeated at all time-points ?
            If False, the index must contain unique values.
            If True, the index must be exactly equal at all time-points.
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
        the name
        whether the index is repeating
    """
    if time_list is not None:
        if not isCollection(time_list):
            raise TypeError(f"Unexpected type '{type(time_list)}' for 'time_list', should be a collection of "
                            f"time-points.")

    # TODO : test for when data is None but other parameters are given
    if data is None:
        if time_list is not None:
            if index is None:
                index = np.arange(len(time_list))

            if (lt := len(time_list)) != (li := len(index)):
                raise ValueError(f"Length of 'time_list' ({lt}) did not match length of 'index' ({li}).")

        if time_col_name is not None:
            warn("No data supplied, 'time_col_name' parameter is ignored.")

        return None, \
            np.empty((0, 0)), \
            np.empty((0, 0)), \
            np.empty(0) if time_list is None else np.array(time_list), \
            np.empty(0) if index is None else np.array(index), \
            np.empty(0) if columns_numerical is None else np.array(columns_numerical), \
            np.empty(0) if columns_string is None else np.array(columns_string),\
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), \
            None, \
            str(name), \
            False

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        numerical_array, string_array, timepoints_array, index, columns_numerical, columns_string, time_col_name = \
            parse_data_df(data.copy(), index, columns_numerical, columns_string, time_list, time_col_name)

        if repeating_index:
            unique_timepoints = np.unique(timepoints_array)

            first_index = index[timepoints_array == unique_timepoints[0]]

            for tp in unique_timepoints[1:]:
                index_tp = index[timepoints_array == tp]

                if not len(first_index) == len(index_tp) or not np.all(first_index == index_tp):
                    raise ValueError(f"Index at time-point {tp} is not equal to index at time-point "
                                     f"{unique_timepoints[0]}.")

        else:
            if not len(index) == len(np.unique(index)):
                raise ValueError("Index values must be all unique.")

        return None, numerical_array, string_array, timepoints_array, index, columns_numerical, columns_string, \
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), time_col_name, str(name), \
            repeating_index

    if isinstance(data, (File, Group)):
        if time_list is not None:
            warn("Data loaded from a H5 file, 'time_list' parameter is ignored.")

        if time_col_name is not None:
            warn("Data loaded from a H5 file, 'time_col_name' parameter is ignored.")

        if repeating_index is not None:
            warn("Data loaded from a H5 file, 'repeating_index' parameter is ignored.")

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
    def sort_and_get_tp(col_name: str,
                        timepoints: Collection) -> np.ndarray:
        data[col_name] = list(map(TimePoint, timepoints))
        data.sort_values(by=col_name, inplace=True, kind='mergesort')

        time_values_ = data[col_name].values
        del data[col_name]

        return time_values_

    # parse INDEX -------------------------------------------------------------
    if index is not None:
        if (l := len(index)) != (s := data.shape[0]):
            raise ValueError(f"'index' parameter has incorrect length {l}, expected {s}.")

        data.index = index

    # parse TIMEPOINTS COLUMN -------------------------------------------------
    if time_list is None:
        if time_col_name is None:
            # set time values to default value '0h'
            time_values = np.array([TimePoint('0h') for _ in enumerate(data.index)])

        else:
            # pick time values from df
            if time_col_name not in data.columns:
                raise ValueError(f"'time_col_name' ('{time_col_name}') is not in the data's columns.")

            # convert column to time-points and sort data in ascending order
            time_values = sort_and_get_tp(time_col_name, data[time_col_name])

    else:
        # pick time values from time_list
        if time_col_name is not None:
            warn("'time_list' parameter already supplied, 'time_col_name' parameter is ignored.")

        if (l := len(time_list)) != (li := len(data.index)):
            raise ValueError(f"'time_list' parameter has incorrect length {l}, expected {li}.")

        # also sort data and time-points in ascending order
        time_values = sort_and_get_tp('__TDF_TMP_COLUMN__', time_list)

    numerical_df = data.select_dtypes(include='number')
    string_df = data.select_dtypes(exclude='number')

    # parse COLUMNS NUM -------------------------------------------------------
    if columns_numerical is not None:
        if (l := len(columns_numerical)) != (s := numerical_df.shape[1]):
            raise ValueError(f"'columns_numerical' parameter has incorrect length {l}, expected {s}.")

        columns_numerical_ = np.array(columns_numerical)

    else:
        columns_numerical_ = np.array(numerical_df.columns.values)

    # parse COLUMNS STR -------------------------------------------------------
    if columns_string is not None:
        if (l := len(columns_string)) != (s := string_df.shape[1]):
            raise ValueError(f"'columns_string' parameter has incorrect length {l}, expected {s}.")

        columns_string_ = np.array(columns_string)

    else:
        columns_string_ = np.array(string_df.columns.values)

    # parse ARRAY NUM ---------------------------------------------------------
    numerical_array = numerical_df.values.copy()
    # enforce 'float' data type
    numerical_array = numerical_array.astype(float)

    # parse ARRAY STR ---------------------------------------------------------
    string_array = string_df.values.copy()
    # enforce 'string' data type
    string_array = string_array.astype(str)

    return numerical_array, string_array, time_values, np.array(data.index), \
        columns_numerical_, columns_string_, time_col_name


def parse_data_h5(data: H5Data,
                  index: Optional[Collection],
                  columns_numerical: Optional[Collection],
                  columns_string: Optional[Collection],
                  lock: Optional[tuple[bool, bool]],
                  name: str) \
        -> tuple[H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, H5Data, tuple[bool, bool], Optional[str], str, bool]:
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

    timepoints_columns_name = None if data.attrs['timepoints_column_name'] == '__TDF_None__' else data.attrs[
        'timepoints_column_name']

    if name != 'No_Name':
        data.attrs['name'] = str(name)

    return data, numerical_array_file, string_array_file, timepoints_array_file, index_file, columns_num_file, \
        columns_str_file, (data.attrs['locked_indices'], data.attrs['locked_columns']), timepoints_columns_name, \
        data.attrs['name'], data.attrs['repeating_index']
