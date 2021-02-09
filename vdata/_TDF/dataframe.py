# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, Union, Optional, Collection, Tuple, Any, List, IO
from typing_extensions import Literal

from vdata.NameUtils import DType, PreSlicer, TimePointList
from vdata.utils import TimePoint, repr_array, repr_index, isCollection, to_tp_list, \
    reformat_index, match_time_points, unique_in_list, trim_time_points
from .NameUtils import TemporalDataFrame_internal_attributes, TemporalDataFrame_reserved_keys
from .base import BaseTemporalDataFrame, copy_TemporalDataFrame
from .views.dataframe import ViewTemporalDataFrame
from .._IO import generalLogger
from .._IO.errors import VValueError, VTypeError, ShapeError

# TODO : remove match_time_points


# ====================================================
# code
def parse_index_and_time_points(_index: Optional[Collection],
                                _data: Optional[Union[Dict, pd.DataFrame]],
                                _time_list: TimePointList,
                                _time_col: Optional[str],
                                _time_points: Optional[List],
                                _columns: Collection[str]) \
        -> Tuple[Dict[TimePoint, pd.DataFrame], List[TimePoint], Optional[str], pd.Index]:
    """
    Given the index, data, time list, time points and columns parameters from a TemporalDataFrame, infer correct
    data DataFrames, get the list of time points, the optional column where time points can be found and the columns
    of the data DataFrames.

    :param _index: index for the dataframe's rows.
    :param _data: data to store as a dataframe.
    :param _time_list: time points for the dataframe's rows.
    :param _time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
        given.
    :param _time_points: a list of time points that should exist.
    :param _columns: column labels.

    :return: data, time_points, time_points_col, data columns
    """
    generalLogger.debug("\t\u23BE Parse index and time points : begin ---------------------------------------- ")

    tp_col = None

    if _data is None:
        generalLogger.debug("\t'data' was not found.")

        if _time_col is not None:
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

                _data = {tp: pd.DataFrame(columns=_columns) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {_time_list}.")

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

                _data = {tp: pd.DataFrame(index=_index[tp], columns=_columns) for tp in _time_points}

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

                _data = {tp: pd.DataFrame(index=_index, columns=_columns) for tp in _time_points}

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
                        generalLogger.warning(f"\tTime points {repr_array(excluded_elements)} were found in 'time_list'"
                                              f"parameter but not in 'time_points'. They will be ignored.")

                    if not len(_index) == len(_time_list):
                        raise ShapeError("Lengths of 'index' and 'time_list' parameters do not match.")

                _time_list = np.array(_time_list)

                _data = {tp: pd.DataFrame(index=np.array(_index)[match_time_points(_time_list, [tp])],
                                          columns=_columns) for tp in
                         _time_points}

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

            _data = pd.DataFrame(_data, columns=_columns)

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        # data is a pandas DataFrame
        else:
            data_len = len(_data)

            generalLogger.debug(f"\tFound data in a DataFrame with {data_len} rows.")

        # check column names are valid
        for key in TemporalDataFrame_reserved_keys:
            if key in _data.columns:
                raise VValueError(f"'{key}' column is reserved and cannot be used in 'data'.")

        # get time list from time_col if possible
        if _time_col is not None:
            if _time_list is None:
                if _time_col in _data.columns:
                    generalLogger.info(f"\tUsing '{_time_col}' as time points data.")

                    _data[_time_col] = to_tp_list(_data[_time_col])
                    _time_list = _data[_time_col]

                    tp_col = _time_col

                else:
                    raise VValueError(f"'{_time_col}' could not be found in 'data'.")

            else:
                generalLogger.warning("\tBoth 'time_list' and 'time_col' parameters were set, 'time_col' will be "
                                      "ignored.")

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
                    _data = {TimePoint('0'): pd.DataFrame(_data, columns=_columns)}

                else:
                    # data and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    _data = {tp: pd.DataFrame(_data, columns=_columns) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if _time_points is None:
                    # data and time list
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # data, time list and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                    _time_list, excluded_elements = trim_time_points(_time_list, _time_points)
                    if len(excluded_elements):
                        generalLogger.warning(f"\tTime points {excluded_elements} were found in 'time_list'"
                                              f"parameter but not in 'time_points'. They will be ignored.")

                        # remove undesired time points from the data
                        _data = _data[match_time_points(_data[tp_col], _time_points)]

                        generalLogger.debug(f"\tNew data has {len(_data)} rows.")

                if data_len != len(_time_list):
                    if data_len * len(_time_points) == len(_time_list):
                        _data = {tp: pd.DataFrame(_data, columns=_columns) for tp in _time_points}

                    else:
                        raise ShapeError("Length of 'time_list' and number of rows in 'data' do not match.")

                else:
                    _data = {tp: pd.DataFrame(_data.loc[match_time_points(_time_list, [tp])], columns=_columns)
                             for tp in _time_points}

        else:
            generalLogger.debug(f"\t\t'index' is : {repr_array(_index)}.")

            if _time_list is None:
                generalLogger.debug("\t\t\t'time_list' was not found.")

                # if data_len != len(_index):
                #     raise ShapeError(f"Length of 'index' and number of rows in 'data' do not match.")

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
                        _data = {tp: pd.DataFrame(_data[tp_index * len_index: (tp_index + 1) * len_index],
                                                  columns=_columns)
                                 for tp_index, tp in enumerate(_time_points)}

                        for tp in _time_points:
                            _data[tp].index = _index

                    else:
                        raise ShapeError("Length of 'index' and number of rows in 'data' do not match.")

                else:
                    _data.index = _index
                    _data = {tp: pd.DataFrame(_data, columns=_columns) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if data_len != len(_time_list):
                    raise ShapeError("Length of 'time_list' and number of rows in 'data' do not match.")

                if _time_points is None:
                    # data, index and time list
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # data, index, time list and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                _time_points = sorted(_time_points)

                if len(_index) != data_len:
                    if len(_index) * len(_time_points) == data_len:
                        _index = np.concatenate([_index for _ in _time_points])

                    else:
                        raise ShapeError("Length of 'index' and number of rows in 'data' do not match.")

                #
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
                        columns=_tmp_columns)
                        for tp_i, tp in enumerate(_time_points)}

                else:
                    _data['__tmp_time__'] = _expanded_time_list
                    _data.sort_values(by='__tmp_time__', inplace=True)

                    _data = {tp: pd.DataFrame(_data[_data['__tmp_time__'] == tp], columns=_tmp_columns)
                             for tp in _time_points}

    if _columns is not None:
        _columns = pd.Index(_columns)

    elif len(_time_points):
        _columns = _data[_time_points[0]].columns
    else:
        _columns = pd.Index([])

    generalLogger.debug(f"\tSet 'time_points' to : {repr_array(_time_points)}.")
    generalLogger.debug(f"\tSet 'time_points_column' to : {tp_col}.")
    generalLogger.debug(f"\tSet 'columns' to : {repr_array(_columns)}.")

    generalLogger.debug("\t\u23BF Parse index and time points : end ------------------------------------------ ")
    return _data, _time_points, tp_col, _columns


class TemporalDataFrame(BaseTemporalDataFrame):
    """
    An extension to pandas DataFrames to include a notion of time on the rows.
    An hidden column '__TPID' contains for each row the list of time points this row appears in.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """

    __base_repr_str = 'TemporalDataFrame'

    def __init__(self, data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_list: Optional[Union[Collection, DType, Literal['*'], TimePoint]] = None,
                 time_col: Optional[str] = None,
                 time_points: Optional[Collection[Union[DType, TimePoint]]] = None,
                 index: Optional[Collection] = None,
                 columns: Optional[Collection] = None,
                 dtype: Optional[DType] = None,
                 name: Optional[Any] = None):
        """
        :param data: data to store as a dataframe.
        :param time_list: time points for the dataframe's rows. The value indicates at which time point a given row
            exists in the dataframe.
            It can be :
                - a collection of values of the same length as the number of rows.
                - a single value to set for all rows.

            In any case, the values can be :
                - a single time point (indicating that the row only exists at that given time point)
                - a collection of time points (indicating that the row exists at all those time points)
                - the character '*' (indicating that the row exists at all time points)

        :param time_col: if time points are not given explicitly with the 'time_list' parameter, a column name can be
            given. This column will be used as the time data.
        :param time_points: a list of time points that should exist. This is useful when using the '*' character to
            specify the list of time points that the TemporalDataFrame should cover.
        :param index: index for the dataframe's rows.
        :param columns: column labels.
        :param dtype: data type to force.
        :param name: optional TemporalDataFrame's name.
        """
        self.name = str(name) if name is not None else 'No_Name'

        generalLogger.debug(f"\u23BE TemporalDataFrame '{self.name}' creation : begin "
                            f"---------------------------------------- ")

        self._time_points = sorted(to_tp_list(unique_in_list(time_points))) if time_points is not None else None
        if self._time_points is not None:
            generalLogger.debug(f"User has defined time points : {repr_array(self._time_points)}.")

        time_list = to_tp_list(time_list, self._time_points) if time_list is not None else None

        # ---------------------------------------------------------------------
        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            data, self._time_points, self._time_points_column_name, self._columns = \
                parse_index_and_time_points(index, None, time_list, time_col, self._time_points, columns)

            self._df = data

        # ---------------------------------------------------------------------
        # data given
        elif isinstance(data, (dict, pd.DataFrame)):

            if isinstance(data, pd.DataFrame):
                # work on a copy of the data to avoid undesired modifications
                data = data.copy()

            data, self._time_points, self._time_points_column_name, self._columns = \
                parse_index_and_time_points(index, data, time_list, time_col, self._time_points, columns)

            generalLogger.debug("Storing data in TemporalDataFrame.")
            self._df = data

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
            repr_str = f"{TemporalDataFrame.__base_repr_str} '{self.name}'\n"

        else:
            repr_str = f"Empty {TemporalDataFrame.__base_repr_str} '{self.name}'\n" \

        # represent at most 6 time points
        if len(self.time_points) > 6:
            for TP in self.time_points[:3]:
                repr_str += f"\033[4mTime point : {repr(TP)}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

            repr_str += f"\nSkipped time points {repr_array(self.time_points[3:-3])} ...\n\n\n"

            for TP in self.time_points[-3:]:
                repr_str += f"\033[4mTime point : {repr(TP)}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

        else:
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {repr(TP)}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

        return repr_str

    def one_TP_repr(self, time_point: TimePoint, n: Optional[int] = None, func: Literal['head', 'tail'] = 'head'):
        """
        Representation of a single time point in this TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :param func: the name of the function to use to limit the output ('head' or 'tail')
        :return: a representation of a single time point in this TemporalDataFrame object
        """
        return repr(self._df[time_point][self.columns].__getattr__(func)(n=n))

    def __getitem__(self, index: Union[PreSlicer,
                                       Tuple[PreSlicer],
                                       Tuple[PreSlicer, PreSlicer],
                                       Tuple[PreSlicer, PreSlicer, PreSlicer]]) \
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

        index = reformat_index(index, self.time_points, self.index, self.columns)

        generalLogger.debug(f'  Refactored index to \n{repr_index(index)}.')

        if not len(index[0]):
            raise VValueError("Time points not found in this TemporalDataFrame.")

        return ViewTemporalDataFrame(self, self._df, index[0], index[1], index[2])

    def __getattribute__(self, attr: str) -> Any:
        """
        Get attribute from this TemporalDataFrame in obj.attr fashion.
        This is called before __getattr__.
        :param attr: an attribute's name to get.
        :return: self.attr
        """
        if attr not in TemporalDataFrame_internal_attributes:
            raise AttributeError

        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :param attr: an attribute's name
        :return: a column with name <attr> from the DataFrame
        """
        if attr in self.columns:
            return self.loc[:, attr]

        else:
            return object.__getattribute__(self, attr)

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

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __asmd_func(self, operation: Literal['__add__', '__sub__', '__mul__', '__truediv__'],
                    value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Common function for modifying all values in this TemporalDataFrame through the common operation (+, -, *, /).
        :param operation: the operation to apply on the TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        _data = self.to_pandas()

        # avoid working on time points
        if self.time_points_column_name is not None:
            _data = _data.loc[:, _data.columns != self.time_points_column_name]

        # add the value to the data
        _data = getattr(_data, operation)(value)

        # insert back the time points
        if self.time_points_column_name is not None:
            _data.insert(list(self.columns).index(self.time_points_column_name), self.time_points_column_name,
                         self.time_points_column)

        time_col = self.time_points_column_name
        time_list = self.time_points_column if time_col is None else None

        return TemporalDataFrame(data=_data,
                                 time_list=time_list,
                                 time_col=time_col,
                                 time_points=self.time_points,
                                 index=self.index,
                                 name=self.name)

    def __add__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        return self.__asmd_func('__add__', value)

    def __sub__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to subtract to values.
        :return: a TemporalDataFrame with new values.
        """
        return self.__asmd_func('__sub__', value)

    def __mul__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to multiply all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self.__asmd_func('__mul__', value)

    def __truediv__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to divide all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self.__asmd_func('__truediv__', value)

    def to_pandas(self, with_time_points: bool = False) -> pd.DataFrame:
        """
        Get the data in a pandas format.
        :param with_time_points: add a column with time points data ?
        :return: the data in a pandas format.
        """
        data = pd.concat([self._df[time_point] for time_point in self.time_points])

        if with_time_points:
            data.insert(0, 'Time_Point', self.time_points_column.values)

        return data

    @property
    def time_points(self) -> List[TimePoint]:
        """
        Get the list of time points in this TemporalDataFrame.
        :return: the list of time points in this TemporalDataFrame.
        """
        return self._time_points

    @property
    def time_points_column_name(self) -> Optional[str]:
        """
        Get the name of the column with time points data. Returns None if '__TPID' is used.
        :return: the name of the column with time points data.
        """
        return self._time_points_column_name

    def index_at(self, time_point: TimePoint) -> pd.Index:
        """
        Get the index of this TemporalDataFrame.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the index of this TemporalDataFrame
        """
        if time_point not in self.time_points:
            raise VValueError(f"TimePoint '{time_point}' cannot be found in this TemporalDataFrame.")

        return self._df[time_point].index

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

    def reindex(self, index: Collection) -> None:
        """
        Conform the index of this TemporalDataFrame to the new index.
        """
        if not isCollection(index):
            raise VTypeError('New index should be an array of values.')

        len_index = self.n_index_total

        if not len(index) == len_index:
            raise VValueError(f"Cannot reindex from an array of length {len(index)}, should be {len_index}.")

        cnt = 0
        for tp in self.time_points:
            self._df[tp] = self._df[tp].reindex(index[cnt:cnt+self.n_index_at(tp)])
            cnt += self.n_index_at(tp)

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this TemporalDataFrame.
        :return: the column names of this TemporalDataFrame
        """
        if self.n_time_points:
            return self._df[self.time_points[0]].columns

        else:
            return self._columns

    @columns.setter
    def columns(self, values: pd.Index) -> None:
        """
        Set the columns of this TemporalDataFrame (except for __TPID).
        :param values: the new column names for this TemporalDataFrame.
        """
        for tp in self._time_points:
            self._df[tp].columns = values

        self._columns = values

    @property
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes in this TemporalDataFrame.
        :return: the dtypes in this TemporalDataFrame.
        """
        if self.n_time_points:
            return self._df[self.time_points[0]].dtypes

        else:
            return pd.Series([])

    def astype(self, dtype: Optional[Union[DType, Dict[str, DType]]]) -> None:
        """
        Cast this TemporalDataFrame to a specified data type.
        :param dtype: a data type.
        """
        if dtype is not None:
            for tp in self._time_points:
                for column in self.columns:
                    if column != self._time_points_column_name:
                        try:
                            self._df[tp][column] = self._df[tp][column].astype(np.float).astype(dtype)

                        except ValueError:
                            pass

    def asColType(self, col_name: str, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        Cast a specific column in this TemporalDataFrame to a specified data type.
        :param col_name: a column name.
        :param dtype: a data type.
        """
        for tp in self._time_points:
            self._df[tp][col_name].astype(dtype)

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a TemporalDataFrame including the index dtype and columns,
        non-null values and memory usage.
        :return: a concise summary of a DataFrame.
        """
        for time_point in self.time_points:
            print(f"\033[4mTime point : {repr(time_point)}\033[0m")
            self._df[time_point].info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                                      null_counts=null_counts)
            print("\n")

    def memory_usage(self, index: bool = True, deep: bool = False) -> pd.Series:
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        if self.n_time_points:
            mem_usage = self._df[self.time_points[0]].memory_usage(index=index, deep=deep)

        else:
            mem_usage = pd.Series([], dtype=object)

        if self.n_time_points > 1:
            for time_point in self.time_points[1:]:
                mem_usage.add(self._df[time_point].memory_usage(index=index, deep=deep))

        return mem_usage

    @property
    def at(self) -> '_VAtIndexer':
        """
        Access a single value for a row/column label pair.
        :return: a single value for a row/column label pair.
        """
        return _VAtIndexer(self, self._df)

    @property
    def iat(self) -> '_ViAtIndexer':
        """
        Access a single value for a row/column pair by integer position.
        :return: a single value for a row/column pair by integer position.
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

    def insert(self, loc, column, value, allow_duplicates=False) -> None:
        """
        Insert column into TemporalDataFrame at specified location.
        :param loc: Insertion index. Must verify 0 <= loc <= len(columns).
        :param column: str, number, or hashable object. Label of the inserted column.
        :param value: int, Series, or array-like
        :param allow_duplicates: duplicate column allowed ?
        """
        for time_point in self.time_points:
            self._df[time_point].insert(loc, column, value, allow_duplicates)

    def copy(self) -> 'TemporalDataFrame':
        """
        Create a new copy of this TemporalDataFrame.
        :return: a copy of this TemporalDataFrame.
        """
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

    def __mean_min_max_func(self, func: Literal['mean', 'min', 'max'], axis) -> Tuple[Dict, np.ndarray, pd.Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {'mean': [self._df[tp][col].__getattr__(func)()
                              for tp in self.time_points for col in self.columns]}
            _time_list = np.repeat(self.time_points, self.n_columns)
            _index = pd.Index(np.concatenate([self.columns for _ in range(self.n_time_points)]))

        elif axis == 1:
            _data = {'mean': [self._df[tp].loc[row].__getattr__(func)()
                              for tp in self.time_points for row in self.index_at(tp)]}
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


class _VAtIndexer:
    """
    Wrapper around pandas _AtIndexer object for use in TemporalDataFrames.
    The .at can access elements by indexing with :
        - a single element (TDF.loc[<element0>])    --> on indexes

    Allowed indexing elements are :
        - a single label
    """

    def __init__(self, parent: 'TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Tuple[Any, Any]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :return: the value stored at the row index and column name.
        """
        return self.__pandas_data.at[key]

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :param value: a value to set.
        """
        row, col = key[0], key[1]
        target_tp = None

        for tp in self.__parent.time_points:
            if row in self.__data[tp].index:
                target_tp = tp
                break

        self.__data[target_tp].at[key] = value


class _ViAtIndexer:
    """
    Wrapper around pandas _iAtIndexer object for use in TemporalDataFrames.
    The .iat can access elements by indexing with :
        - a 2-tuple of elements (TDF.loc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
    """

    def __init__(self, parent: TemporalDataFrame, data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        self.__parent = parent
        self.__data = data
        self.__pandas_data = parent.to_pandas()

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row # and column #
        :return: the value stored at the row # and column #.
        """
        return self.__pandas_data.iat[key]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        row, col = key[0], key[1]
        target_tp = None

        row_cumul = 0
        for tp in self.__parent.time_points:

            if row_cumul + len(self.__data[tp]) >= row:
                target_tp = tp
                break

            else:
                row_cumul += len(self.__data[tp])

        self.__data[target_tp].iat[key[0] - row_cumul, key[1]] = value
