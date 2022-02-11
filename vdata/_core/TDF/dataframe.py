# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Union, Optional, Collection, Any, Literal

from .name_utils import TemporalDataFrame_internal_attributes, TemporalDataFrame_reserved_keys
from .utils import unique_in_list, trim_time_points
from .base import BaseTemporalDataFrame
from .copy import copy_TemporalDataFrame
from .indexers import _VAtIndexer, _ViAtIndexer, _VLocIndexer, _ViLocIndexer
from .views import ViewTemporalDataFrame
from ..name_utils import TimePointList, PreSlicer
from ..utils import match_time_points, to_tp_list, to_list, reformat_index, repr_index
from vdata.name_utils import DType
from vdata.utils import repr_array, isCollection
from vdata.time_point import TimePoint
from ...IO import generalLogger, VValueError, VTypeError, ShapeError, VLockError
from ...h5pickle import Group, File

# TODO : remove match_time_points


# ====================================================
# code
def parse_index_and_time_points(_index: Optional[Collection],
                                _data: Optional[Union[dict, pd.DataFrame]],
                                _time_list: 'TimePointList',
                                _time_col_name: Optional[str],
                                _time_points: Optional[list],
                                _columns: Collection[str]) \
        -> tuple[
            dict['TimePoint', np.ndarray],
            list['TimePoint'],
            Optional[str],
            dict['TimePoint', pd.Index],
            pd.Index
        ]:
    """
    Given the index, data, time list, time points and columns parameters from a TemporalDataFrame, infer correct
    data DataFrames, get the list of time points, the optional column where time points can be found and the columns
    of the data DataFrames.

    :param _index: index for the dataframe's rows.
    :param _data: data to store as a dataframe.
    :param _time_list: time points for the dataframe's rows.
    :param _time_col_name: if time points are not given explicitly with the 'time_list' parameter, a column name can be
        given.
    :param _time_points: a list of time points that should exist.
    :param _columns: column labels.

    :return: data, time_points, time_points_col_name, time_points_col, index and columns
    """
    generalLogger.debug("\t\u23BE Parse index and time points : begin ---------------------------------------- ")

    tp_col_name = None

    if _data is None:
        generalLogger.debug("\t'data' was not found.")

        if _time_col_name is not None:
            generalLogger.warning("\t'time_col' parameter was set but no data was provided, 'time_col' will be "
                                  "ignored.")

        if _index is None:
            generalLogger.debug("\t\t'index' was not found.")

            if _time_list is None:
                generalLogger.debug("\t\t\t'time_list' was not found.")

                if _time_points is None:
                    # nothing provided
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = []

                else:
                    # only list of time points was provided
                    generalLogger.debug(f"\t\t\t\t'time points' is : {repr_array(_time_points)}.")

                _index = {tp: pd.Index([]) for tp in _time_points}
                _data = {tp: np.array([]) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if _time_points is None:
                    # only a time_list was provided
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = sorted(list(unique_in_list(_time_list)))

                else:
                    # time_list and time_points were provided
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    _time_list, excluded_elements = trim_time_points(_time_list, _time_points)
                    if len(excluded_elements):
                        generalLogger.warning(f"\tTime points {excluded_elements} were found in 'time_list'"
                                              f"parameter but not in 'time_points'. They will be ignored.")

                _index = {tp: [] for tp in _time_points}
                current_index = 0

                for item in _time_list:
                    if isCollection(item):
                        for sub_item in item:
                            _index[sub_item].append(current_index)

                    else:
                        _index[item].append(current_index)

                    current_index += 1

                _index = {tp: pd.Index(_index[tp]) for tp in _index}

                if _columns is not None:
                    _data = {tp: np.empty((len(_index[tp]), len(pd.Index(_columns)))) for tp in _time_points}
                    for tp in _time_points:
                        _data[tp][:] = np.nan

                else:
                    _data = {tp: np.array([]) for tp in _time_points}

        else:
            generalLogger.debug(f"\t\t'index' is : {repr_array(_index)}.")

            if _time_list is None:
                generalLogger.debug("\t\t\t'time_list' was not found.")

                if _time_points is None:
                    # only the index was provided
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    generalLogger.info(f"\tSetting all time points to default value '{TimePoint('0')}'.")

                    _time_points = [TimePoint('0')]

                else:
                    # index and time_points were provided
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                _index = {tp: pd.Index(_index) for tp in _time_points}
                _data = {tp: np.array([]) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if _time_points is None:
                    # index and _time_list were provided
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    if not len(_index) == len(_time_list):
                        raise ShapeError(f"Lengths of 'index' ({repr_array(_index)}) and 'time_list' "
                                         f"({repr_array(_time_list)}) parameters do not match.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # index, _time_list and _time_points were provided
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    _time_list, excluded_elements = trim_time_points(_time_list, _time_points)
                    if len(excluded_elements):
                        generalLogger.warning(f"\tTime points {repr_array(excluded_elements)} were found in "
                                              f"'time_list' parameter but not in 'time_points'. They will be ignored.")

                    if not len(_index) == len(_time_list):
                        raise ShapeError("Lengths of 'index' and 'time_list' parameters do not match.")

                _time_list = np.array(_time_list)

                _data = {tp: pd.DataFrame(index=np.array(_index)[match_time_points(_time_list, [tp])],
                                          columns=_columns).values for tp in _time_points}
                _index = {tp: pd.Index(_index)[match_time_points(_time_list, [tp])] for tp in _time_points}

        if _columns is not None:
            _columns = pd.Index(_columns)

        else:
            _columns = pd.Index([])

    else:
        # First, get data length

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        # if data is a dict, check that the dict can be converted to a DataFrame
        if isinstance(_data, dict):
            # get number of rows in data
            data_len = 1
            for value in _data.values():
                value_len = len(value) if isCollection(value) else 1

                if value_len != data_len and data_len != 1 and value_len != 1:
                    raise ShapeError("All items in 'data' must have the same length "
                                     "(or be a unique value to set for all rows).")

                if data_len == 1:
                    data_len = value_len

            generalLogger.debug(f"\tFound data in a dictionary with {data_len} rows.")

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        # data is a pandas DataFrame
        else:
            data_len = len(_data)

            generalLogger.debug(f"\tFound data in a DataFrame with {data_len} rows and {len(_data.columns)} columns.")

        _data = pd.DataFrame(_data, columns=_columns)

        # check column names are valid
        for key in TemporalDataFrame_reserved_keys:
            if key in _data.columns:
                raise VValueError(f"'{key}' column is reserved and cannot be used in 'data'.")

        # get time list from time_col if possible
        if _time_col_name is not None:
            if _time_list is None:
                if _time_col_name in _data.columns:
                    generalLogger.info(f"\tUsing '{_time_col_name}' as time points data.")

                    if _time_points is not None:
                        ref = _time_points

                    else:
                        ref = to_tp_list(unique_in_list(_data[_time_col_name]) - {'*'})

                    _time_list = to_tp_list(_data[_time_col_name], ref)

                    del _data[_time_col_name]

                    tp_col_name = _time_col_name

                else:
                    raise VValueError(f"'{_time_col_name}' could not be found in 'data'.")

            else:
                generalLogger.warning("\tBoth 'time_list' and 'time_col' parameters were set, 'time_col' will be "
                                      "ignored.")

        _columns = _data.columns

        # -----------------------------------------------------------------------------------
        # Then, parse data
        if _index is None:
            generalLogger.debug("\t\t'index' was not found.")

            if _time_list is None:
                generalLogger.debug("\t\t\t'time_list' was not found.")

                if _time_points is None:
                    # only data provided
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    generalLogger.info(f"\tSetting all time points to default value '{TimePoint('0')}'.")

                    _time_points = [TimePoint('0')]
                    _index = {TimePoint('0'): _data.index}
                    _data = {TimePoint('0'): _data.values}

                else:
                    # data and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    _index = {tp: _data.index for tp in _time_points}
                    _data = {tp: _data.values for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if _time_points is None:
                    # data and time list
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # data, time list and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    new_time_list, excluded_elements = trim_time_points(_time_list, _time_points)
                    if len(excluded_elements):
                        generalLogger.warning(f"\tTime points {list(excluded_elements)} were found in 'time_list' "
                                              f"parameter but not in 'time_points'. They will be ignored.")

                        # remove undesired time points from the data
                        _data = _data[match_time_points(_time_list, _time_points)]

                        generalLogger.debug(f"\tNew data has {len(_data)} rows.")

                    _time_list = new_time_list

                if len(_data) != len(_time_list):
                    if len(_data) * len(_time_points) == len(_time_list):
                        _index = {tp: _data.index for tp in _time_points}
                        _data = {tp: _data.values for tp in _time_points}

                    else:
                        raise ShapeError("Length of 'time_list' and number of rows in 'data' do not match.")

                else:
                    _data = {tp: _data.loc[match_time_points(_time_list, [tp])] for tp in _time_points}

                    _index = {tp: _data[tp].index for tp in _time_points}
                    _data = {k: v.values for k, v in _data.items()}

        else:
            generalLogger.debug(f"\t\t'index' is : {repr_array(_index)}.")

            if _time_list is None:
                generalLogger.debug("\t\t\t'time_list' was not found.")

                if _time_points is None:
                    # data and index
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    generalLogger.info(f"\tSetting all time points to default value '{TimePoint('0')}'.")

                    _time_points = [TimePoint('0')]

                else:
                    # data, index and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                len_index = len(_index)

                if len_index != data_len:
                    if len_index * len(_time_points) == data_len:
                        _data = {tp: _data[tp_index * len_index: (tp_index + 1) * len_index]
                                 for tp_index, tp in enumerate(_time_points)}

                        for tp in _time_points:
                            _data[tp].index = _index

                        _index = {tp: pd.Index(_index) for tp in _time_points}
                        _data = {k: v.values for k, v in _data.items()}

                    else:
                        raise ShapeError("Length of 'index' and number of rows in 'data' do not match.")

                else:
                    _data = {tp: _data.values for tp in _time_points}
                    _index = {tp: pd.Index(_index) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if _time_points is None:
                    # data, index and time list
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # data, index, time list and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                _time_points = sorted(_time_points)

                if data_len != len(_time_list):
                    if data_len * len(_time_points) == len(_time_list):
                        _data = pd.concat([_data for _ in range(len(_time_points))])
                        generalLogger.debug(f"Refactored data by concatenating {len(_time_points)} times.")

                    else:
                        raise ShapeError("Length of 'time_list' and number of rows in 'data' do not match.")

                if len(_index) != len(_data):
                    if len(_index) * len(_time_points) == len(_data):
                        _index = pd.Index(np.concatenate([_index for _ in _time_points]))
                        generalLogger.debug(f"Refactored index by concatenating {len(_time_points)} times.")

                    else:
                        raise ShapeError("Length of 'index' and number of rows in 'data' do not match.")

                _tmp_columns = _columns if _columns is not None else _data.columns
                _expanded_time_list = []

                for tp_i, tp in enumerate(_time_list):
                    if isCollection(tp):
                        _data = pd.concat([_data.iloc[:tp_i]] +
                                          [pd.DataFrame(_data.iloc[tp_i]).T for _ in range(len(set(tp))-1)] +
                                          [_data.iloc[tp_i:]], sort=False)

                        for tp_bis in tp:
                            _expanded_time_list.append(tp_bis)

                    else:
                        _expanded_time_list.append(tp)

                _data.index = _index

                if _expanded_time_list == sorted(_expanded_time_list):
                    tp_occurrences = Counter(_expanded_time_list)
                    tp_cumulative_occurrences = [0] + list(np.cumsum([tp_occurrences[tp] for tp in _time_points]))

                    _data = {tp: pd.DataFrame(
                        _data.iloc[tp_cumulative_occurrences[tp_i]:tp_cumulative_occurrences[tp_i + 1]],
                        columns=_tmp_columns) for tp_i, tp in enumerate(_time_points)}

                else:
                    _data['__tmp_time__'] = _expanded_time_list
                    _data.sort_values(by='__tmp_time__', inplace=True)

                    _data = {tp: pd.DataFrame(_data[_data['__tmp_time__'] == tp], columns=_tmp_columns)
                             for tp in _time_points}

                _index = {tp: _data[tp].index for tp in _time_points}
                _data = {k: v.values for k, v in _data.items()}

    generalLogger.debug(f"\tSet 'time_points' to : {repr_array(_time_points)}.")
    generalLogger.debug(f"\tSet 'time_points_column' to : {tp_col_name}.")
    generalLogger.debug(f"\tSet 'columns' to : {repr_array(_columns)}.")

    generalLogger.debug("\t\u23BF Parse index and time points : end ------------------------------------------ ")
    return _data, _time_points, tp_col_name, _index, _columns


class TemporalDataFrame(BaseTemporalDataFrame):
    """
    An equivalent to pandas DataFrames to include a notion of time on the rows.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """
    def __init__(self,
                 data: Optional[Union[dict, pd.DataFrame, Group, File]] = None,
                 time_list: Optional[Union[Collection, 'DType', Literal['*'], 'TimePoint']] = None,
                 time_col_name: Optional[str] = None,
                 time_points: Optional[Collection[Union['DType', 'TimePoint']]] = None,
                 index: Optional[Collection] = None,
                 columns: Optional[Collection] = None,
                 dtype: Optional['DType'] = None,
                 name: Optional[Any] = None,
                 file: Optional[File] = None):
        """
        Args:
            data: data to store as a dataframe.
            time_list: time points for the dataframe's rows. The value indicates at which time point a given row
                exists in the dataframe.
                It can be :
                    - a collection of values of the same length as the number of rows.
                    - a single value to set for all rows.

                In any case, the values can be :
                    - a single time point (indicating that the row only exists at that given time point)
                    - a collection of time points (indicating that the row exists at all those time points)
                    - the character '*' (indicating that the row exists at all time points)

            time_col_name: if time points are not given explicitly with the 'time_list' parameter, a column name can
                be given. This column will be used as the time data.
            :param time_points: a list of time points that should exist. This is useful when using the '*' character to
                specify the list of time points that the TemporalDataFrame should cover.
            index: index for the dataframe's rows.
            columns: column labels.
            dtype: data type to force.
            name: optional TemporalDataFrame's name.
            file: optional TemporalDataFrame's h5 file for backing.
        """
        self._name = str(name) if name is not None else 'No_Name'

        generalLogger.debug(f"\u23BE TemporalDataFrame '{self.name}' creation : begin "
                            f"---------------------------------------- ")

        self._file = None
        self._is_locked = (False, False)

        self._time_points = sorted(
            to_tp_list(unique_in_list(to_list(time_points)), [])) if time_points is not None else None

        if self._time_points is not None:
            generalLogger.debug(f"User has defined time points : {repr_array(self._time_points)}.")

        if time_list is not None:
            if self._time_points is not None:
                ref = self._time_points

            else:
                ref = to_tp_list(unique_in_list(to_list(time_list)) - {'*'})

            time_list = to_tp_list(time_list, ref)

        # ---------------------------------------------------------------------
        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            index = index if index is not None else data.index if data is not None and len(data.index) else None
            columns = columns if columns is not None else data.columns if data is not None and len(
                data.columns) else None

            data, self._time_points, self._time_points_column_name, self._index, self._columns = \
                parse_index_and_time_points(index, None, time_list, time_col_name, self._time_points, columns)

            self._df = data

        # ---------------------------------------------------------------------
        # data given
        elif isinstance(data, (dict, pd.DataFrame)):

            if isinstance(data, pd.DataFrame):
                # work on a copy of the data to avoid undesired modifications
                data = data.copy()

            data, self._time_points, self._time_points_column_name, self._index, self._columns = \
                parse_index_and_time_points(index, data, time_list, time_col_name, self._time_points, columns)

            generalLogger.debug("Storing data in TemporalDataFrame.")
            self._df = data

            if file is not None:
                self._file = file

        # ---------------------------------------------------------------------
        # data from hdf5 file
        elif isinstance(data, (Group, File)):

            assert index is not None, "'index' parameter must be set when reading an h5 file."
            assert columns is not None, "'columns' parameter must be set when reading an h5 file."

            self._columns = pd.Index(columns)
            self._time_points_column_name = str(time_col_name) if time_col_name is not None else None

            self._time_points = []
            self._index = {}
            self._df = {}

            self.__load_from_file(data, index)

        # ---------------------------------------------------------------------
        # invalid data
        else:
            raise VTypeError(f"Type {type(data)} is not handled for 'data' parameter.")

        # ---------------------------------------------------------------------
        # cast to correct dtype
        self.astype(dtype)

        # get list of time points that can be found in the DataFrame
        self._time_points = sorted(self._time_points)

        generalLogger.debug(f"Found time points {repr_array(self._time_points)} from stored DataFrame.")

        generalLogger.debug(f"Shape is : {self.shape}.")

        generalLogger.debug(f"\u23BF TemporalDataFrame '{self.name}' creation : end "
                            f"------------------------------------------ ")

    def __repr__(self) -> str:
        """
        Description for this TemporalDataFrame object to print.
        :return: a description of this TemporalDataFrame object
        """
        if not self.empty:
            repr_str = f"{'Backed ' if self.is_backed else ''}TemporalDataFrame '{self.name}'\n"

        else:
            repr_str = f"Empty {'backed ' if self.is_backed else ''}TemporalDataFrame '{self.name}'\n"

        repr_str += self._head()

        return repr_str

    def __getitem__(self, index: Union['PreSlicer',
                                       tuple['PreSlicer'],
                                       tuple['PreSlicer', 'PreSlicer'],
                                       tuple['PreSlicer', 'PreSlicer', 'PreSlicer']]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a view from this TemporalDataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index, a 2-tuple or a 3-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Indexes are converted to a 3-tuple :
                * TDF[index]            --> (index, :, :)
                * TDF[index1, index2]   --> (index1, index2, :)

            The first element in the 3-tuple is the list of time points to select, the second element is a
            collection of rows to select, the third element is a collection of columns to select.

            The values ':' or '...' are shortcuts for 'take all values'.

            Example:
                * TemporalDataFrame[:] or TemporalDataFrame[...]    --> select all data
                * TemporalDataFrame[0]                              --> select all data from time point 0
                * TemporalDataFrame[[0, 1], [1, 2, 3], 'col1']      --> select data from time points 0 and 1 for rows
                                                                    with index in [1, 2, 3] and column 'col1'
        :return: a view on a sub-set of a TemporalDataFrame
        """
        generalLogger.debug(f"TemporalDataFrame '{self.name}' sub-setting - - - - - - - - - - - - - - ")
        generalLogger.debug(f'  Got index \n{repr_index(index)}.')

        if isinstance(index, tuple) and len(index) == 3 and not isCollection(index[2]) \
                and not isinstance(index[2], slice) and index[2] is not ... \
                and (index[0] is ... or (isinstance(index[0], slice) and index[0] == slice(None)))\
                and (index[1] is ... or index[1] == slice(None)):
            return self.__getattr__(index[2])

        else:
            index = reformat_index(index, self.time_points, self.index, self.columns)

            generalLogger.debug(f'  Refactored index to \n{repr_index(index)}.')

            if not len(index[0]):
                raise VValueError("Time points not found in this TemporalDataFrame.")

            return ViewTemporalDataFrame(self, self._df, index[0], index[1], index[2], self._is_locked[0])

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :param attr: an attribute's name
        :return: a column with name <attr> from the DataFrame
        """
        try:
            return object.__getattribute__(self, str(attr))

        except AttributeError:
            try:
                return object.__getattribute__(self, 'loc')[:, attr]

            except KeyError:
                raise AttributeError(f"'TemporalDataFrame' object has no attribute '{attr}'")

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set value for a regular attribute of for a column in the DataFrame.
        :param attr: an attribute's name
        :param value: a value to be set into the attribute
        """
        if attr in TemporalDataFrame_internal_attributes:
            object.__setattr__(self, attr, value)

        elif attr in self.columns:
            self.loc[:, attr] = value

            if self.is_backed and self._file.file.mode == 'r+':
                self.write()

        else:
            self.insert(self.n_columns, attr, value)

    def __delattr__(self, col: str) -> None:
        if self.is_locked[1]:
            raise VLockError("Cannot use 'delattr' functionality on a locked TemporalDataFrame.")

        else:
            if col in self.columns:
                col_index = np.where(self.columns == col)[0][0]

                for time_point in self.time_points:
                    self._df[time_point] = np.delete(self._df[time_point], col_index, 1)

                self._columns = self._columns.delete(col_index)

                if self.is_backed and self._file.file.mode == 'r+':
                    self.write()

            else:
                raise VValueError(f"Column '{col}' not found in TemporalDataFrame '{self.name}'.")

    def __add__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__add__', value)

    def __sub__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to subtract to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__sub__', value)

    def __mul__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to multiply all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__mul__', value)

    def __truediv__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to divide all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__truediv__', value)

    def __eq__(self, other):
        if isinstance(other, (TemporalDataFrame, ViewTemporalDataFrame)):
            return self.time_points == other.time_points and self.index == other.index and self.columns == \
                   other.columns and all([self._df[tp] == other._df[tp] for tp in self.time_points])

        elif self.n_columns == 1:
            return self.eq(other).values.flatten()

        else:
            return self.eq(other)

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state) -> None:
        self.__dict__ = state

    def __load_from_file(self,
                         file: Union[Group, File],
                         index: Collection) -> None:
        """

        """
        self._time_points = sorted([TimePoint(tp) for tp in file['data'].keys()])
        self._df = {}
        _index = {}

        for time_point in self._time_points:
            self._df[time_point] = file['data'][str(time_point)]

            _index[time_point] = pd.Index(index[:len(self._df[time_point])])
            index = index[len(self._df[time_point]):]

        self._index = _index
        self._file = file

    @property
    def is_backed(self) -> bool:
        """
        Is this TemporalDataFrame backed on an h5 file ?
        :return: Is this TemporalDataFrame backed on an h5 file ?
        """
        return object.__getattribute__(self, '_file') is not None

    @property
    def file(self) -> Union[File, Group]:
        """
        Get the h5 file this TemporalDataFrame is backed on.
        :return: the h5 file this TemporalDataFrame is backed on.
        """
        return object.__getattribute__(self, '_file')

    @file.setter
    def file(self, new_file: Union[File, Group]):
        """
        Set the h5 file this TemporalDataFrame is backed on.
        :param new_file: an h5 file to back this TemporalDataFrame.
        """
        if not isinstance(new_file, (File, Group)):
            raise VTypeError(f"Cannot back TemporalDataFrame '{self.name}' with an object of type '{type(new_file)}'.")

        self._file = new_file

    @property
    def is_locked(self) -> tuple[bool, bool]:
        """
        Get this TemporalDataFrame's lock.
        This controls what can be modified with 2 boolean values :
            1. True --> cannot use self.index.setter() and self.reindex()
            2. True --> cannot use self.__delattr__(), self.columns.setter() and self.insert()
        """
        return object.__getattribute__(self, '_is_locked')

    def lock(self, values: tuple[bool, bool]) -> None:
        """
        Set this TemporalDataFrame's lock.
        This controls what can be modified with 2 boolean values :
            1. True --> cannot use self.index.setter() and self.reindex()
            2. True --> cannot use self.__delattr__(), self.columns.setter() and self.insert()
        """
        if not isinstance(values, tuple) or len(values) != 2 or not isinstance(values[0], bool) or not isinstance(
                values[1], bool):
            raise VValueError("'values' must be a 2-tuple of boolean values.")

        self._is_locked = values

    def to_pandas(self, with_time_points: bool = False) -> pd.DataFrame:
        """
        Get the data in a pandas format.
        :param with_time_points: add a column with time points data ?
        :return: the data in a pandas format.
        """
        if object.__getattribute__(self, 'is_backed'):
            data = pd.concat([pd.DataFrame(self._df[time_point][()],
                                           index=self.index_at(time_point),
                                           columns=self.columns)
                              for time_point in self.time_points])

        else:
            data = pd.concat([pd.DataFrame(self._df[time_point],
                                           index=self.index_at(time_point),
                                           columns=self.columns)
                              for time_point in self.time_points])

        if with_time_points:
            if self.time_points_column_name is not None:
                data.insert(0, self.time_points_column_name, self.time_points_column)

            else:
                data.insert(0, 'Time_Point', self.time_points_column)

        return data

    @property
    def time_points(self) -> list['TimePoint']:
        """
        Get the list of time points in this TemporalDataFrame.
        :return: the list of time points in this TemporalDataFrame.
        """
        return object.__getattribute__(self, '_time_points')

    @property
    def time_points_column_name(self) -> Optional[str]:
        """
        Get the name of the column with time points data. Returns None if no column is used.
        :return: the name of the column with time points data.
        """
        return object.__getattribute__(self, '_time_points_column_name')

    @time_points_column_name.setter
    def time_points_column_name(self, value: str) -> None:
        """
        Set the name of the column with time points data.
        :param value: a new name for the column with time points data.
        """
        value = str(value)

        self._time_points_column_name = value

        if self.is_backed and self.file.file.mode == 'r+':
            self.file['time_col_name'][()] = value
            self.file.file.flush()

    def index_at(self, time_point: Union['TimePoint', str]) -> pd.Index:
        """
        Get the index of this TemporalDataFrame.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the index of this TemporalDataFrame
        """
        if not isinstance(time_point, TimePoint):
            time_point = TimePoint(time_point)

        if time_point not in self.time_points:
            raise VValueError(f"TimePoint '{time_point}' cannot be found in this TemporalDataFrame.")

        return self._index[time_point]

    @property
    def index(self) -> pd.Index:
        """
        Get the full index of this TemporalDataFrame (concatenated over all time points).
        :return: the full index of this TemporalDataFrame.
        """
        _index = pd.Index([])

        for time_point in self.time_points:
            _index = _index.append(self.index_at(time_point))

        return _index

    @index.setter
    def index(self, values: Collection) -> None:
        """
        Set a new index for observations.
        :param values: collection of new index values.
        """
        if self.is_locked[0]:
            raise VLockError("Cannot use 'index.setter' functionality on a locked TemporalDataFrame.")

        else:
            if not isCollection(values):
                raise VTypeError('New index should be an array of values.')

            len_index = self.n_index_total

            if not len(values) == len_index:
                raise VValueError(f"Cannot reindex from an array of length {len(values)}, should be {len_index}.")

            cnt = 0
            for tp in self.time_points:
                self._index[tp] = pd.Index(values[cnt:cnt+self.n_index_at(tp)])
                cnt += self.n_index_at(tp)

            if self.is_backed and self._file.file.mode == 'r+':
                self._file['index'][()] = self.index
                self._file.file.flush()

    def reindex(self, index: Collection) -> None:
        """
        Conform the index of this TemporalDataFrame to the new index.
        :param index: a new index with same elements as in the current index but the order can be different.
        """
        if self.is_locked[0]:
            raise VLockError("Cannot use 'reindex' functionality on a locked TemporalDataFrame.")

        else:
            if not isCollection(index):
                raise VTypeError('New index should be an array of values.')

            index = pd.Index(index)
            len_index = self.n_index_total

            if not len(index) == len_index:
                raise VValueError(f"Cannot reindex from an array of length {len(index)}, should be {len_index}.")

            cnt = 0
            for tp in self.time_points:
                index_for_tp = index[cnt:cnt + self.n_index_at(tp)]

                if not all(self.index_at(tp).isin(index_for_tp)):
                    raise VValueError(f"Some values in the new index are not present in the current index "
                                      f"at time point '{tp}'.")

                index_loc = [self.index_at(tp).get_loc(e) for e in index_for_tp]

                # change row order based on new index
                self._df[tp] = self._df[tp][index_loc]

                # modify index
                self._index[tp] = index_for_tp

                cnt += self.n_index_at(tp)

            if self.is_backed and self._file.file.mode == 'r+':
                self._file['index'][()] = self.index
                self._file.file.flush()

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this TemporalDataFrame.
        :return: the column names of this TemporalDataFrame
        """
        return self._columns

    @columns.setter
    def columns(self, values: pd.Index) -> None:
        """
        Set the columns of this TemporalDataFrame (except for __TPID).
        :param values: the new column names for this TemporalDataFrame.
        """
        if self.is_locked[1]:
            raise VLockError("Cannot use 'columns.setter' functionality on a locked TemporalDataFrame.")

        else:
            values = pd.Index(values)
            if not len(values) == len(self._columns):
                raise VValueError(f"Expected an Index of len {len(self._columns)} for 'values' parameter.")

            self._columns = values

            if self.is_backed and self._file.file.mode == 'r+':
                self.write()

    @property
    def name(self) -> str:
        """
        Get the name of this TemporalDataFrame.
        :return: the name of this TemporalDataFrame.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set a new name for this TemporalDataFrame.
        :param value: the new name for this TemporalDataFrame.
        """
        self._name = value

    @property
    def dtype(self) -> Optional[np.dtype]:
        """
        Return the dtype of this TemporalDataFrame.
        :return: the dtype of this TemporalDataFrame.
        """
        if self.n_time_points:
            return self._df[self.time_points[0]].dtype

        else:
            return None

    def astype(self, dtype: Optional['DType']) -> None:
        """
        Cast this TemporalDataFrame to a specified data type.
        :param dtype: a data type.
        """
        if dtype is not None and dtype != self.dtype:
            if self.is_backed:
                generalLogger.warning("Cannot set data type for a backed TemporalDataFrame.\n"
                                      "If you really want to modify the data type, create a copy of this "
                                      "TemporalDataFrame before doing so.")

            else:
                for tp in self._time_points:
                    self._df[tp] = self._df[tp].astype(dtype)

    @property
    def at(self) -> '_VAtIndexer':
        """
        Access a single value from a row/column label pair.
        :return: a single value from a row/column label pair.
        """
        return _VAtIndexer(self, self._df)

    @property
    def iat(self) -> '_ViAtIndexer':
        """
        Access a single value from a row/column pair by integer position.
        :return: a single value from a row/column pair by integer position.
        """
        return _ViAtIndexer(self, self._df)

    @property
    def loc(self) -> '_VLocIndexer':
        """
        Access a group of rows and columns by label(s) or a boolean array.

        Allowed inputs are:
            - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an
            integer position along the index).
            - A list or array of labels, e.g. ['a', 'b', 'c'].
            - A slice object with labels, e.g. 'a':'f'.
            - A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
            - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
            for indexing (one of the above)

        :return: a group of rows and columns
        """
        return _VLocIndexer(self, self._df)

    @property
    def iloc(self) -> '_ViLocIndexer':
        """
        Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).

        Allowed inputs are:
            - An integer, e.g. 5.
            - A list or array of integers, e.g. [4, 3, 0].
            - A slice object with ints, e.g. 1:7.
            - A boolean array.
            - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
            for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the
            calling object, but would like to base your selection on some value.

        :return: a group of rows and columns
        """
        return _ViLocIndexer(self, self._df)

    def insert(self, loc: int, column: str, values: Any) -> None:
        """
        Insert column into TemporalDataFrame at specified location.
        :param loc: Insertion index. Must verify 0 <= loc <= len(columns).
        :param column: str, number, or hashable object. Label of the inserted column.
        :param values: int, Series, or array-like
        """
        def _insert(arr, _values):
            try:
                arr = np.insert(arr, loc, _values, axis=1)

            except ValueError:
                arr = arr.astype(object)
                arr = np.insert(arr, loc, _values, axis=1)

            return arr

        if self.is_locked[1]:
            raise VLockError("Cannot use 'insert' functionality on a locked TemporalDataFrame.")

        else:
            if isCollection(values):
                if self.n_index_total != len(values):
                    raise VValueError("Length of values does not match length of index.")

                cumul = 0
                for time_point in self.time_points:
                    values_to_insert = values[cumul:cumul + self.n_index_at(time_point)]

                    # insert values into array
                    self._df[time_point] = _insert(self._df[time_point], values_to_insert)

                    cumul += self.n_index_at(time_point)

            else:
                for time_point in self.time_points:
                    # insert values into array
                    self._df[time_point] = _insert(self._df[time_point], values)

            # insert column name into column index
            self._columns = self._columns.insert(loc, column)

            if self.is_backed and self._file.file.mode == 'r+':
                self.write()

    def copy(self) -> 'TemporalDataFrame':
        """
        Create a new copy of this TemporalDataFrame.
        :return: a copy of this TemporalDataFrame.
        """
        if self.is_backed:
            return copy_TemporalDataFrame(self)

        else:
            return copy_TemporalDataFrame(self)

    def to_csv(self, path: Union[str, Path], sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True) -> None:
        """
        Save this TemporalDataFrame in a csv file.
        :param path: a path to the csv file.
        :param sep: String of length 1. Field delimiter for the output file.
        :param na_rep: Missing data representation.
        :param index: Write row names (index) ?
        :param header: Write out the column names ? If a list of strings is given it is
            assumed to be aliases for the column names.
        """
        # save DataFrame to csv
        self.to_pandas(with_time_points=True).to_csv(path, sep=sep, na_rep=na_rep, index=index, header=header)

    def __mean_min_max_func(self, func: Literal['mean', 'min', 'max'],
                            axis: Literal[0, 1]) \
            -> tuple[
                dict[Literal['mean', 'min', 'max'], np.ndarray],
                np.ndarray,
                pd.Index
            ]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {func: np.concatenate([getattr(np.array(self._df[tp]), func)(axis=0) for tp in self.time_points])}
            _time_list = np.repeat(self.time_points, self.n_columns)
            _index = pd.Index(np.concatenate([self.columns for _ in range(self.n_time_points)]))

        elif axis == 1:
            _data = {func: np.concatenate([getattr(np.array(self._df[tp]), func)(axis=1) for tp in self.time_points])}
            _time_list = self.time_points_column
            _index = self.index

        else:
            raise VValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    def mean(self, axis: Literal[0, 1] = 0) -> 'TemporalDataFrame':
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('mean', axis)

        _name = f"Mean of {self.name}" if self.name != 'No_Name' else None
        return TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def min(self, axis: Literal[0, 1] = 0) -> 'TemporalDataFrame':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self.name}" if self.name != 'No_Name' else None
        return TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def max(self, axis: Literal[0, 1] = 0) -> 'TemporalDataFrame':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self.name}" if self.name != 'No_Name' else None
        return TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def merge(self, other: 'TemporalDataFrame', name: Optional[str] = None) -> 'TemporalDataFrame':
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.
        :param other: a TemporalDataFrame to merge with this one.
        :param name: a name for the merged TemporalDataFrame.
        :return: a new merged TemporalDataFrame.
        """
        if not self.time_points == other.time_points:
            raise VValueError("Cannot merge TemporalDataFrames with different time points.")

        if not self.columns.equals(other.columns):
            raise VValueError("Cannot merge TemporalDataFrames with different columns.")

        if not self.time_points_column_name == other.time_points_column_name:
            raise VValueError("Cannot merge TemporalDataFrames with different 'time_col' parameter values.")

        if self.empty:
            combined_index = np.array([])
            for tp in self.time_points:
                combined_index = np.concatenate((combined_index,
                                                 self.index_at(tp).values,
                                                 other.index_at(tp).values))

            _data = pd.DataFrame(index=combined_index, columns=self.columns)

        else:
            _data = pd.DataFrame(columns=self.columns)

            for time_point in self.time_points:
                if any(other.index_at(time_point).isin(self.index_at(time_point))):
                    raise VValueError(f"TemporalDataFrames to merge have index values in common at time point "
                                      f"'{time_point}'.")

                _data = pd.concat((_data, self[time_point].to_pandas(), other[time_point].to_pandas()))

            _data.columns = _data.columns.astype(self.columns.dtype)

        if self.time_points_column_name is None:
            _time_list = [time_point for time_point in self.time_points
                          for _ in range(self.n_index_at(time_point) + other.n_index_at(time_point))]

        else:
            _time_list = None

        return TemporalDataFrame(data=_data, time_list=_time_list, time_col_name=self.time_points_column_name,
                                 name=name)

    def write(self, file: Optional[Union[str, Path, Group, File]] = None) -> None:
        """
        Save this TemporalDataFrame in HDF5 file format.

        Args:
            file: path to save the TemporalDataFrame.
        """
        from ..._read_write import write_TemporalDataFrame

        if file is None:
            save_file = self._file.parent
            write_TemporalDataFrame(self, save_file, self.name)
            self._file.file.flush()

        elif isinstance(file, (Group, File)):
            save_file = file
            write_TemporalDataFrame(self, save_file, self.name)

        else:
            save_file = File(file, 'w')
            write_TemporalDataFrame(self, save_file, self.name)

        if not self.is_backed:
            index = np.concatenate([i.values for i in self._index.values()])
            self.__load_from_file(save_file[self.name], index)
