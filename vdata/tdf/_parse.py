# coding: utf-8
# Created on 29/03/2022 11:43
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from typing import Any, Collection
from warnings import warn

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
import pandas as pd

from vdata._typing import IFS, NDArray_IFS
from vdata.names import NO_NAME
from vdata.timepoint import TimePoint, TimePointArray
from vdata.utils import isCollection, obj_as_str


# ====================================================
# code
class TimedArray:
    
    def __init__(self,
                 arr: npt.NDArray[Any] | None,
                 time: TimePointArray | None,
                 repeating_index: bool):
        if arr is None and isinstance(time, TimePointArray):
            self.arr = np.arange(len(time))
            self.time = time
            
        elif arr is not None and time is None:
            self.arr = arr
            self.time = np.repeat(TimePoint('0h'), len(arr))                # type: ignore[call-overload]
            
        elif isinstance(arr, np.ndarray) and isinstance(time, TimePointArray):
            if len(arr) != len(time):
                raise ValueError(f"Length of 'time_list' ({len(time)}) did not match length of 'index' " \
                                 f"({len(arr)}).")
                
            self.arr = arr
            self.time = time
                
        elif arr is None and time is None:
            self.arr = np.empty(0, dtype=np.float64)
            self.time = TimePointArray(np.empty(0, dtype=object))
            
        else:
            raise NotImplementedError
        
        if repeating_index:
            unique_timepoints = np.unique(self.time)

            first_index = self.arr[self.time == unique_timepoints[0]]

            for tp in unique_timepoints[1:]:
                index_tp = self.arr[self.time == tp]

                if not len(first_index) == len(index_tp) or not np.all(first_index == index_tp):
                    raise ValueError(f"Index at time-point {tp} is not equal to index at time-point "
                                     f"{unique_timepoints[0]}.")

        elif len(self.arr) != len(np.unique(self.arr)):
            raise ValueError("Index values must be all unique.")
        
        self.repeating_index = repeating_index
        
        
def _sort_and_get_tp(data: pd.DataFrame | None, 
                     col_name: str | None,
                     timepoints: TimePointArray) -> TimePointArray:
    if data is None:
        return timepoints
    
    if col_name is None:
        col_name = '__TDF_TMP_COLUMN__'
    
    _dtype = data.columns.dtype
    data[col_name] = timepoints
    data.sort_values(by=col_name, inplace=True, kind='mergesort')

    time_values_ = TimePointArray(list(data[col_name]))
    del data[col_name]
    data.columns = data.columns.astype(_dtype)

    return time_values_


def _get_timed_index(index: Collection[IFS] | None,
                     time_list: Collection[IFS | TimePoint] | IFS | TimePoint | None,
                     time_col_name: str | None,
                     data: pd.DataFrame | None,
                     repeating_index: bool) -> TimedArray:
    if isinstance(data, pd.DataFrame) and index is not None:
        data.index = pd.Index(index)
    
    if time_list is None and time_col_name is not None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'time_col_name' parameter was given without data.")
        
        if time_col_name not in data.columns:
            raise ValueError(f"'time_col_name' ('{time_col_name}') is not in the data's columns.")

        _time_list = _sort_and_get_tp(data, time_col_name, TimePointArray(data[time_col_name]))
    
    elif time_list is not None:
        if time_col_name is not None:
            warn("'time_list' parameter already supplied, 'time_col_name' parameter is ignored.")
        
        if isCollection(time_list):
            _time_list = _sort_and_get_tp(data, time_col_name, TimePointArray(time_list))
            
        elif isinstance(time_list, (TimePoint, float, int, str, np.int_, np.float_)):
            _time_list = TimePointArray([TimePoint(time_list)])
            
        else:
            raise TypeError(f"Unexpected type '{type(time_list)}' for 'time_list', should be a collection of "
                            f"time-points.")
            
    else:
        _time_list = None
        
    if isinstance(data, pd.DataFrame):
        _index = data.index.values
        
    elif index is not None:
        _index = np.array(index)
        
    else:
        _index = None
    
    return TimedArray(_index, _time_list, repeating_index)
            

def parse_data_h5(data: ch.H5Dict[Any],
                  lock: tuple[bool, bool] | None,
                  name: str) -> ch.AttributeManager:
    if lock is not None:
        data.attributes['locked_indices'], data.attributes['locked_columns'] = bool(lock[0]), bool(lock[1])

    if name != NO_NAME:
        data.attributes['name'] = str(name)

    return data.attributes


def parse_data(data: dict[str, NDArray_IFS] | pd.DataFrame | None,
               index: Collection[IFS] | None,
               repeating_index: bool,
               columns: Collection[IFS] | None,
               time_list: Collection[IFS | TimePoint] | IFS | TimePoint | None,
               time_col_name: str | None,
               lock: tuple[bool, bool] | None,
               name: str) -> tuple[
                   npt.NDArray[np.int_ | np.float_], 
                   npt.NDArray[np.str_],
                   TimePointArray,
                   NDArray_IFS,
                   NDArray_IFS,
                   NDArray_IFS,
                   tuple[bool, bool],
                   str | None,
                   str,
                   bool
                ]:
    """
    Parse the user-given data to create a TemporalDataFrame.

    Args:
        data: Optional object containing the data to store in this TemporalDataFrame. It can be :
            - a dictionary of ['column_name': [values]], where [values] has always the same length
            - a pandas DataFrame
            - a single value to fill the data with
        index: Optional indices to set or to substitute to indices defined in data if data is a pandas DataFrame.
        repeating_index: Is the index repeated at all time-points ?
            If False, the index must contain unique values.
            If True, the index must be exactly equal at all time-points.
        columns: Optional column names.
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
    if isinstance(data, dict):
        data = pd.DataFrame(data)
        
    if data is not None:
        data = data.copy()
    
    timed_index = _get_timed_index(index, time_list, time_col_name, data, repeating_index)
    
    # TODO : test for when data is None but other parameters are given
    if data is None:
        if time_col_name is not None:
            warn("No data supplied, 'time_col_name' parameter is ignored.")

        return np.empty((len(timed_index.arr), 
                         0 if columns is None else len(columns))), \
            np.empty((len(timed_index.arr), 0), dtype=str), \
            timed_index.time, \
            timed_index.arr, \
            np.empty(0) if columns is None else np.array(columns), \
            np.empty(0),\
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), \
            None, \
            str(name), \
            timed_index.repeating_index

    if isinstance(data, pd.DataFrame):
        numerical_array, string_array, columns_numerical, columns_string = \
            parse_data_df(data, columns)

        return numerical_array, \
            string_array, \
            timed_index.time,\
            obj_as_str(np.array(data.index)),\
            columns_numerical, \
            columns_string, \
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), \
            time_col_name, \
            str(name), \
            timed_index.repeating_index

    if not isCollection(data):
        if timed_index.arr is not None and columns is not None:
            numerical_array = np.empty((len(timed_index.arr), len(columns)), dtype=float)
            numerical_array[:] = np.nan

        else:
            numerical_array = np.empty((0, 0))

        if timed_index.arr is not None:
            string_array = np.empty((len(timed_index.arr), 0), dtype=str)
            string_array[:] = np.nan

        else:
            string_array = np.empty((0, 0), dtype=str)

        return numerical_array, \
            string_array, \
            timed_index.time, \
            timed_index.arr, \
            np.empty(0) if columns is None else np.array(columns), \
            np.empty(0), \
            (False, False) if lock is None else (bool(lock[0]), bool(lock[1])), \
            time_col_name, \
            str(name), \
            timed_index.repeating_index

    raise ValueError("Invalid 'data' supplied for creating a TemporalDataFrame. 'data' can be :\n"
                     " - a dictionary of ['column_name': [values]], where [values] has always the same length \n"
                     " - a pandas DataFrame \n"
                     " - a H5 File or Group containing numerical and string data")
    

def parse_data_df(data: pd.DataFrame,
                  columns: Collection[IFS] | None) -> tuple[
                      npt.NDArray[np.float64], 
                      npt.NDArray[np.str_], 
                      NDArray_IFS,
                      NDArray_IFS
                  ]:
    numerical_df = data.select_dtypes(include='number')
    string_df = data.select_dtypes(exclude='number')
    
    if columns is not None:
        columns_numerical = np.array(columns)[:numerical_df.shape[1]]
        columns_string = np.array(columns)[numerical_df.shape[1]:]
        
    else:
        columns_numerical = np.array(numerical_df.columns.values)
        columns_string = np.array(string_df.columns.values)

    # parse ARRAY NUM ---------------------------------------------------------
    # enforce 'float' data type
    numerical_array = numerical_df.values.astype(float)

    # parse ARRAY STR ---------------------------------------------------------
    # enforce 'string' data type
    string_array = string_df.values.astype(str)

    return numerical_array, string_array, obj_as_str(columns_numerical), obj_as_str(columns_string)
