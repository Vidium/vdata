# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, Collection, Tuple, Any, List, IO, Hashable, Iterable
from typing_extensions import Literal

from vdata.NameUtils import DType, PreSlicer
from vdata.utils import TimePoint, repr_array, repr_index, isCollection, to_tp_list, to_tp_tuple, to_list, \
    reformat_index, match_time_points, unique_in_list, trim_time_points
from .NameUtils import TemporalDataFrame_internal_attributes, TemporalDataFrame_reserved_keys
from .views.dataframe import ViewTemporalDataFrame
from .._IO import generalLogger
from .._IO.errors import VValueError, VTypeError, ShapeError


# ====================================================
# code
def parse_index_and_time_points(_index: Optional[Collection],
                                _data: Optional[Union[Dict, pd.DataFrame]],
                                _time_list: Optional[Tuple],
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
    generalLogger.debug(f"\t\u23BE Parse index and time points : begin ---------------------------------------- ")

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
                        raise ShapeError(f"Lengths of 'index' and 'time_list' parameters do not match.")

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

                    _data[_time_col] = to_tp_tuple(_data[_time_col])
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

                if data_len != len(_time_list):
                    raise ShapeError(f"Length of 'time_list' and number of rows in 'data' do not match.")

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

                _time_list = np.array(_time_list)

                _data = {tp: pd.DataFrame(_data.loc[match_time_points(_time_list, [tp])], columns=_columns) for tp in
                         _time_points}

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
                        raise ShapeError(f"Length of 'index' and number of rows in 'data' do not match.")

                else:
                    _data.index = _index
                    _data = {tp: pd.DataFrame(_data, columns=_columns) for tp in _time_points}

            else:
                generalLogger.debug(f"\t\t\t'time_list' is : {repr_array(_time_list)}.")

                if data_len != len(_time_list):
                    raise ShapeError(f"Length of 'time_list' and number of rows in 'data' do not match.")

                if _time_points is None:
                    # data, index and time list
                    generalLogger.debug("\t\t\t\t'time_points' was not found.")

                    _time_points = list(unique_in_list(_time_list))

                else:
                    # data, index, time list and time points
                    generalLogger.debug(f"\t\t\t\t'time_points' is : {repr_array(_time_points)}.")

                if len(_index) != data_len:
                    if len(_index) * len(_time_points) == data_len:
                        _index = np.concatenate([_index for _ in _time_points])

                    else:
                        raise ShapeError(f"Length of 'index' and number of rows in 'data' do not match.")

                _data.index = _index
                _data = {tp: pd.DataFrame(_data.loc[match_time_points(_time_list, [tp])],
                                          columns=_columns) for tp in _time_points}

    if _columns is not None:
        _columns = pd.Index(_columns)

    elif len(_time_points):
        _columns = _data[_time_points[0]].columns
    else:
        _columns = pd.Index([])

    generalLogger.debug(f"\tSet 'time_points' to : {repr_array(_time_points)}.")
    generalLogger.debug(f"\tSet 'time_points_column' to : {tp_col}.")
    generalLogger.debug(f"\tSet 'columns' to : {repr_array(_columns)}.")

    generalLogger.debug(f"\t\u23BF Parse index and time points : end ------------------------------------------ ")
    return _data, _time_points, tp_col, _columns


class TemporalDataFrame:
    """
    An extension to pandas DataFrames to include a notion of time on the rows.
    An hidden column '__TPID' contains for each row the list of time points this row appears in.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """

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

        time_list = to_tp_tuple(time_list, self._time_points) if time_list is not None else None

        # ---------------------------------------------------------------------
        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            data, self._time_points, self._time_points_col, self._columns = \
                parse_index_and_time_points(index, None, time_list, time_col, self._time_points, columns)

            self._df = data

        # ---------------------------------------------------------------------
        # data given
        elif isinstance(data, (dict, pd.DataFrame)):

            if isinstance(data, pd.DataFrame):
                # work on a copy of the data to avoid undesired modifications
                data = data.copy()

            data, self._time_points, self._time_points_col, self._columns = \
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
            repr_str = f"TemporalDataFrame '{self.name}'\n"

        else:
            repr_str = f"Empty TemporalDataFrame '{self.name}'\n" \

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
        TODO : update docstring !
        Get a view from the DataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index or a 2-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Single indexes are converted to a 2-tuple :
                * single index --> (index, :)

            The first element in the 2-tuple is the list of time points to select, the second element is a
            collection of bool that can be obtained from conditions on the TemporalDataFrame as it is done with
            pandas DataFrrepr_indexames.

            The values ':' or '...' are shortcuts for 'take all values'.

            Example:
                * TemporalDataFrame[:] or TemporalDataFrame[...]    --> select all data
                * TemporalDataFrame[0]                              --> select all data from time point 0
                * TemporalDataFrame[[0, 1], <condition>]            --> select all data from time points 0 and 1 which
                                                                        match the condition. <condition> takes the form
                                                                        of a list of booleans indicating the rows to
                                                                        select with 'True'.
        :return: a view on a sub-set of a TemporalDataFrame
        """
        generalLogger.debug(f"TemporalDataFrame '{self.name}' sub-setting - - - - - - - - - - - - - - ")
        generalLogger.debug(f'  Got index \n{repr_index(index)}.')

        index = reformat_index(index, self.time_points, self.index, self.columns)

        generalLogger.debug(f'  Refactored index to \n{repr_index(index)}.')

        if not len(index[0]):
            raise VValueError("Time points not found in this TemporalDataFrame.")

        return ViewTemporalDataFrame(self, self._df, index[0], index[1], index[2])

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    df: Union[pd.DataFrame, 'TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        """
        Set values in the DataFrame with a DataFrame.
        The columns and the number of rows must match.
        :param index: a sub-setting index. (see __getitem__ for more details)
        :param df: a DataFrame with values to set.
        """
        self[index].set(df)

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
            return self._df[attr]

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
            self._df[attr] = value

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __len__(self) -> int:
        """
        Returns the length of info axis.
        :return: the length of info axis.
        """
        return self.n_index

    def __add__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
        time_list = self._df['__TPID'] if time_col is None else None

        return TemporalDataFrame(self.to_pandas() + value,
                                 time_list=time_list,
                                 time_col=time_col,
                                 time_points=self.time_points,
                                 index=self.index,
                                 name=self.name)

    def __sub__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to subtract to values.
        :return: a TemporalDataFrame with new values.
        """
        time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
        time_list = self._df['__TPID'] if time_col is None else None

        return TemporalDataFrame(self.to_pandas() - value,
                                 time_list=time_list,
                                 time_col=time_col,
                                 time_points=self.time_points,
                                 index=self.index,
                                 name=self.name)

    def __mul__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to multiply all values by.
        :return: a TemporalDataFrame with new values.
        """
        time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
        time_list = self._df['__TPID'] if time_col is None else None

        return TemporalDataFrame(self.to_pandas() * value,
                                 time_list=time_list,
                                 time_col=time_col,
                                 time_points=self.time_points,
                                 index=self.index,
                                 name=self.name)

    def __truediv__(self, value: Union[int, float]) -> 'TemporalDataFrame':
        """
        Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to divide all values by.
        :return: a TemporalDataFrame with new values.
        """
        time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
        time_list = self._df['__TPID'] if time_col is None else None

        return TemporalDataFrame(self.to_pandas() / value,
                                 time_list=time_list,
                                 time_col=time_col,
                                 time_points=self.time_points,
                                 index=self.index,
                                 name=self.name)

    def to_pandas(self) -> Any:
        """
        Get the data in a pandas format.
        :return: the data in a pandas format.
        """
        columns = self.columns[0] if len(self.columns) == 1 else self.columns

        return self._df[columns]

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
        return self._time_points_col if self._time_points_col != '__TPID' else None

    @property
    def time_points_column(self) -> pd.Series:
        """
        Get the time points data for all rows in this TemporalDataFrame.
        :return: the time points data.
        """
        _data = pd.Series([])

        for time_point in self.time_points:
            if self._time_points_col is not None:
                _data = pd.concat((_data, self._df[time_point][self._time_points_col]))

            else:
                _data = pd.concat((_data, pd.Series([time_point for _ in range(self.n_index_at(time_point))])))

        return _data

    @property
    def n_time_points(self) -> int:
        """
        Get the number of distinct time points in this TemporalDataFrame.
        :return: the number of time points.
        """
        return len(self.time_points)

    def index_at(self, time_point: TimePoint) -> pd.Index:
        """
        Get the index of this TemporalDataFrame.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the index of this TemporalDataFrame
        """
        return self._df[time_point].index

    def n_index_at(self, time_point: TimePoint) -> int:
        """
        Get the length of the index at a given time point.
        :param time_point: a time point in this TemporalDataFrame.
        :return: the length of the index at a given time point.
        """
        return len(self._df[time_point].index)

    @property
    def index(self) -> pd.Index:
        """TODO"""
        _index = pd.Index([])

        for time_point in self.time_points:
            _index = _index.union(self.index_at(time_point), sort=False)

        return _index

    @property
    def n_index_total(self) -> int:
        """
        Get the number of indexes.
        :return: the number of indexes.
        """
        return sum([self.n_index_at(TP) for TP in self.time_points])

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this TemporalDataFrame (but mask the reserved __TPID column).
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
    def n_columns(self) -> int:
        """
        Get the number of columns in this TemporalDataFrame
        :return: the number of columns
        """
        return len(self._columns)

    @property
    def dtypes(self) -> pd.Series:
        """
        Return the dtypes in this TemporalDataFrame.
        :return: the dtypes in this TemporalDataFrame.
        """
        if self.n_time_points:
            return self._df[self.time_points[0]][self.columns].dtypes

        else:
            return pd.Series([])

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        Cast this TemporalDataFrame to a specified data type.
        :param dtype: a data type.
        """
        for tp in self._time_points:
            for column in self.columns:
                if column != self._time_points_col:
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

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of this TemporalDataFrame.
        :return: a Numpy representation of the DataFrame.
        """
        return self._df[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self._df[self.columns].axes

    @property
    def ndim(self) -> Literal[3]:
        """
        Return an int representing the number of axes / array dimensions.
        :return: 3
        """
        return 3

    @property
    def size(self) -> int:
        """
        Return the number of rows times number of columns.
        :return: an int representing the number of elements in this object.
        """
        return self._df[self.columns].size

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of this TemporalDataFrame
        (nb_time_points, [n_index_at(time point) for all time points], nb_col).
        :return: a tuple representing the dimensionality of this TemporalDataFrame
        """
        return self.n_time_points, [self.n_index_at(TP) for TP in self.time_points], self.n_columns

    def memory_usage(self, index: bool = True, deep: bool = False) -> Dict[TimePoint, pd.Series]:
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        mem_dict = {}
        for time_point in self.time_points:
            mem_dict[time_point] = self._df[time_point].memory_usage(index=index, deep=deep)

        return mem_dict

    @property
    def empty(self) -> bool:
        """
        Indicator whether this TemporalDataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        if not self.n_time_points or not self.n_columns or not self.n_index_total:
            return True

        return False

    def head(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the first n rows for the object based on position.

        For negative values of n, this function returns all rows except the last n rows.
        :return: the first n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n)}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    def tail(self, n: int = 5, time_points: PreSlicer = slice(None, None, None)) -> str:
        """
        This function returns the last n rows for the object based on position.

        For negative values of n, this function returns all rows except the first n rows.
        :return: the last n rows.
        """
        sub_TDF = self[time_points]

        if sub_TDF.n_time_points:
            repr_str = ""
            for TP in sub_TDF.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{sub_TDF[TP].one_TP_repr(TP, n, func='tail')}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in sub_TDF.columns]}\n" \
                       f"Index: {[idx for idx in sub_TDF.index]}"

        return repr_str

    @property
    def at(self) -> '_VAtIndexer':
        """
        Access a single value for a row/column label pair.
        :return: a single value for a row/column label pair.
        """
        return _VAtIndexer(self)

    @property
    def iat(self) -> '_ViAtIndexer':
        """
        Access a single value for a row/column pair by integer position.
        :return: a single value for a row/column pair by integer position.
        """
        return _ViAtIndexer(self)

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
            for indexing (one of the above) TODO

        :return: a group of rows and columns
        """
        return _VLocIndexer(self)

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
        return _ViLocIndexer(self)

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

    def items(self) -> Dict[TimePoint, List[Tuple[Optional[Hashable], pd.Series]]]:
        """
        Iterate over (column name, Series) pairs.
        :return: a tuple with the column name and the content as a Series.
        """
        for column in self.columns:
            yield column, pd.concat((self._df[time_point][column] for time_point in self.time_points))

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        if self.n_time_points:
            return list(self._df[self.time_points[0]].keys())

        else:
            return []

    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'TemporalDataFrame':
        """
        Whether each element in the TemporalDataFrame is contained in values.
        :return: whether each element in the DataFrame is contained in values.
        """
        if self._time_points_col == '__TPID':
            time_points = self._df['__TPID'].tolist()
            time_col = None

        else:
            time_points = self._df[self._time_points_col].tolist()
            time_col = self._time_points_col

        return TemporalDataFrame(self._df.isin(values)[self.columns], time_points=time_points, time_col=time_col)

    def eq(self, other: Any, axis: Literal[0, 1, 'index', 'column'] = 'columns',
           level: Any = None) -> 'TemporalDataFrame':
        """
        Get Equal to of TemporalDataFrame and other, element-wise (binary operator eq).
        Equivalent to '=='.
        :param other: Any single or multiple element data structure, or list-like object.
        :param axis: {0 or ‘index’, 1 or ‘columns’}
        :param level: int or label
        """
        if self._time_points_col == '__TPID':
            time_points = self._df['__TPID'].tolist()
            time_col = None

        else:
            time_points = self._df[self._time_points_col].tolist()
            time_col = self._time_points_col

        return TemporalDataFrame(self._df.eq(other, axis, level)[self.columns],
                                 time_points=time_points, time_col=time_col)

    def copy(self) -> 'TemporalDataFrame':
        """
        Create a new copy of this TemporalDataFrame.
        :return: a copy of this TemporalDataFrame.
        """
        time_points = self.df_data[self._time_points_col].copy() if self._time_points_col == '__TPID' else None
        time_col = self._time_points_col if self._time_points_col != '__TPID' else None
        return TemporalDataFrame(self.df_data[self.columns].copy(),
                                 time_points=time_points, time_col=time_col,
                                 index=self.index.copy(), columns=self.columns.copy())

    def to_csv(self, path: Union[str, Path], sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True) -> None:
        """TODO"""
        # get full DataFrame and rename the '__TPID' column to 'Time_Point'
        data_to_save = self.df_data
        data_to_save.rename(columns={'__TPID': 'Time_Point'}, inplace=True)

        # save DataFrame to csv
        data_to_save.to_csv(path, sep=sep, na_rep=na_rep, index=index, header=header)


# TODO :
class _VLocIndexer:
    """
    Wrapper around pandas _LocIndexer object for use in TemporalDataFrames.
    """

    def __init__(self, parent: TemporalDataFrame):
        """
        :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
        """
        self.__parent = parent
        self.__loc = parent.df_data.loc

    def __getLoc(self, key: Union[Any, Tuple[Any, Any]]) -> Any:
        """
        Parse the key and get un-formatted data from the parent TemporalDataFrame.
        :param key: loc index.
        :return: pandas DataFrame, Series or a single value.
        """
        generalLogger.debug(u'\u23BE .loc access : begin ------------------------------------------------------- ')
        if isinstance(key, tuple):
            generalLogger.debug(f'Key is a tuple : ({key[0]}, {key[1]})')
            if isinstance(key[1], slice):
                generalLogger.debug(f'Second item is a slice : checking ...')
                if key[1].start not in self.__parent.columns:
                    raise VValueError(f"Key '{key[1].start}' was not found in the column names.")

                elif key[1].stop not in self.__parent.columns:
                    raise VValueError(f"Key '{key[1].stop}' was not found in the column names.")
                generalLogger.debug(f'... OK')

            else:
                # collection of column names
                if isinstance(key[1], (tuple, list, pd.Index)):
                    # prevent access to reserved columns
                    for k in to_list(key[1]):
                        if k not in self.__parent.columns:
                            raise VValueError(f"Key '{k}' was not found in the column names.")

                    # also get __TPID for getting time point data
                    key = (key[0], ['__TPID'] + list(key[1]))

                # single column name
                else:
                    # also get __TPID for getting time point data
                    key = (key[0], ['__TPID', key[1]])

        result = self.__loc[key]
        generalLogger.debug(f'.loc data is : \n{result}')
        generalLogger.debug(u'\u23BF .loc access : end --------------------------------------------------------- ')

        return result

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> Any:
        """
        Get rows and columns from the loc.
        :param key: loc index.
        :return: TemporalDataFrame or single value.
        """
        result = self.__getLoc(key)

        if isinstance(result, pd.DataFrame):
            if result.shape == (1, 2):
                generalLogger.debug(f'.loc data is a DataFrame (a single value).')

                return result.iat[0, 1]

            else:
                generalLogger.debug(f'.loc data is a DataFrame (multiple rows or columns).')

                tp_slicer = set(result['__TPID'])

                return ViewTemporalDataFrame(self.__parent, tp_slicer, result.index, result.columns[1:])

        elif isinstance(result, pd.Series):
            if len(result) == 1:
                generalLogger.debug(f'.loc data is a Series (a single value).')

                return result.values[0]

            if len(result) == 2 and '__TPID' in result.index:
                generalLogger.debug(f'.loc data is a Series (a single value).')

                return result.iat[1]

            else:
                generalLogger.debug(f'.loc data is a Series (a row).')

                tp_slicer = result['__TPID']

                result = result.to_frame().T
                return ViewTemporalDataFrame(self.__parent, tp_slicer, result.index, result.columns[1:])

        else:
            generalLogger.debug(f'.loc data is a single value.')

            return result

    def __setitem__(self, key: Union[Any, Tuple[Any, Any]], value: Any) -> None:
        """
        Set rows and columns from the loc.
        :param key: loc index.
        :param value: pandas DataFrame, Series or a single value to set.
        """
        # This as no real effect, it is done to check that the key exists in the view.
        _ = self.__getLoc(key)

        if isinstance(value, TemporalDataFrame):
            value = value.df_data[value.columns]

        # Actually set a value at the (index, column) key.
        self.__parent.df_data.loc[key] = value


class _ViLocIndexer:
    """
    Wrapper around pandas _iLocIndexer object for use in TemporalDataFrames.
    """

    def __init__(self, parent: TemporalDataFrame):
        """
        :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
        """
        self.__parent = parent
        self.__iloc = parent.df_data.iloc

    def __getiLoc(self, key: Union[Any, Tuple[Any, Any]]) -> Any:
        """
        Parse the key and get un-formatted data from the parent TemporalDataFrame.
        :param key: iloc index.
        :return: pandas DataFrame, Series or a single value.
        """
        generalLogger.debug(u'\u23BE .iloc access : begin ------------------------------------------------------ ')
        if isinstance(key, tuple):
            generalLogger.debug(f'Key is a tuple : ({key[0]}, {key[1]})')
            if isinstance(key[1], slice):
                generalLogger.debug(f'Second item is a slice : update it.')

                start = key[1].start + 1 if key[1].start is not None else 1
                stop = key[1].stop + 1 if key[1].stop is not None else len(self.__parent.columns) + 1
                step = key[1].step if key[1].step is not None else 1

                new_key = [0] + list(range(start, stop, step))

                key = (key[0], new_key)

            elif isinstance(key[1], int):
                generalLogger.debug(f'Second item is an int : update it.')
                if isinstance(key[0], (list, np.ndarray, slice)):
                    key = (key[0], [0, key[1] + 1])

                else:
                    key = (key[0], key[1] + 1)

            elif isinstance(key[1], (list, np.ndarray)) and len(key[1]):
                if isinstance(key[1][0], bool):
                    generalLogger.debug(f'Second item is an array of bool : update it.')
                    key = (key[0], [True] + key[1])

                elif isinstance(key[1][0], int):
                    generalLogger.debug(f'Second item is an array of int : update it.')
                    key = (key[0], [0] + [v + 1 for v in key[1]])

        result = self.__iloc[key]
        generalLogger.debug(f'.iloc data is : \n{result}')
        generalLogger.debug(u'\u23BF .iloc access : end -------------------------------------------------------- ')

        return result

    def __getitem__(self, key: Union[Any, Tuple[Any, Any]]) -> Any:
        """
        Get rows and columns from the loc.
        :param key: loc index.
        :return: TemporalDataFrame or single value.
        """
        result = self.__getiLoc(key)

        if isinstance(result, pd.DataFrame):
            if result.shape == (1, 2):
                generalLogger.debug(f'.loc data is a DataFrame (a single value).')

                return result.iat[0, 1]

            else:
                generalLogger.debug(f'.loc data is a DataFrame (multiple rows or columns).')

                tp_slicer = sorted(set(result['__TPID']))
                generalLogger.debug(f'tp slicer is {tp_slicer}.')

                return ViewTemporalDataFrame(self.__parent, tp_slicer, result.index, result.columns[1:])

        elif isinstance(result, pd.Series):
            pass
            if len(result) == 1 and '__TPID' not in result.index:
                generalLogger.debug(f'.loc data is a Series (a single value).')

                return result.values[0]

            if len(result) == 2 and '__TPID' in result.index:
                generalLogger.debug(f'.loc data is a Series (a single value).')

                return result.iat[1]

            else:
                generalLogger.debug(f'.loc data is a Series (a row).')

                tp_slicer = result['__TPID']

                result = result.to_frame().T
                return ViewTemporalDataFrame(self.__parent, tp_slicer, result.index, result.columns[1:])

        else:
            generalLogger.debug(f'.loc data is a single value.')

            return result

    def __setitem__(self, key: Union[Any, Tuple[Any, Any]], value: Any) -> None:
        """
        Set rows and columns from the loc.
        :param key: loc index.
        :param value: pandas DataFrame, Series or a single value to set.
        """
        # TODO : debug
        # This as no real effect, it is done to check that the key exists in the view.
        _ = self.__getiLoc(key)

        if isinstance(value, TemporalDataFrame):
            value = value.df_data[value.columns]

        # Actually set a value at the (index, column) key.
        self.__parent.df_data.iloc[key] = value


# class _ViLocIndexer:
#     """
#     A simple wrapper around the pandas _iLocIndexer object.
#     """
#
#     def __init__(self, iloc: _iLocIndexer):
#         """
#         :param iloc: a pandas _iLocIndexer.
#         """
#         self.iloc = iloc
#
#     def __getitem__(self, index: Any) -> Any:
#         """
#         Get rows and columns from the iloc.
#         :param index: an index for getting rows and columns.
#         :return: a TemporalDataFrame built from the rows and columns accessed from the loc.
#         """
#         value = self.iloc[index]
#
#         if isinstance(value, pd.Series):
#             return value[value.keys()[1:]]
#
#         return value
#
#     # def __setitem__(self, index: Any, value: Any) -> None:
#     #     """
#     #     TODO
#     #     """
#     #     self.iloc[index] = value


class _VAtIndexer:
    """
    Wrapper around pandas _AtIndexer object for use in TemporalDataFrames.
    """

    def __init__(self, parent: TemporalDataFrame):
        """
        :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
        """
        self.__parent = parent
        self.__at = parent.df_data.at

    def __getitem__(self, key: Tuple[Any, Any]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :return: the value stored at the row index and column name.
        """
        # prevent access to reserved columns
        if key[1] not in self.__parent.columns:
            raise VValueError(f"Key '{key[1]}' was not found in the column names.")
        return self.__at[key]

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :param value: a value to set.
        """
        # prevent access to reserved columns
        if key[1] not in self.__parent.columns:
            raise VValueError(f"Key '{key[1]}' was not found in the column names.")
        # This as no real effect, it is done to check that the key (index, column) exists in the view.
        self.__at[key] = value
        # Actually set a value at the (index, column) key.
        self.__parent.df_data.at[key] = value


class _ViAtIndexer:
    """
    Wrapper around pandas _iAtIndexer object for use in TemporalDataFrames.
    """

    def __init__(self, parent: TemporalDataFrame):
        """
        :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
        """
        self.__parent = parent
        self.__iat = parent.df_data.iat

    def __getitem__(self, key: Tuple[int, int]) -> Any:
        """
        Get values using the _AtIndexer.
        :param key: a tuple of row # and column #
        :return: the value stored at the row # and column #.
        """
        # increment key[1] by 1 because '__TPID' is masked
        return self.__iat[key[0], key[1] + 1]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        # increment key[1] by 1 because '__TPID' is masked
        # This as no real effect, it is done to check that the key (index, column) exists in the view.
        self.__iat[key[0], key[1] + 1] = value
        # Actually set a value at the (index, column) key.
        generalLogger.debug(f"Setting value '{value}' at row '{key[0]}' and col '{key[1] + 1}'")
        self.__parent.df_data.iat[key[0], key[1] + 1] = value

# class _ViewVAtIndexer:
#     """
#     Wrapper around pandas _AtIndexer object for use in ViewTemporalDataFrames.
#     """
#     def __init__(self, parent: TemporalDataFrame):
#         """
#         :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
#         """
#         self.__parent = parent
#         self.__at = parent.df_data.at
#
#     def __getitem__(self, key: Tuple[Any, Any]) -> Any:
#         """
#         Get values using the _AtIndexer.
#         :param key: a tuple of row index and column name.
#         :return: the value stored at the row index and column name.
#         """
#         return self.__at[key]
#
#     def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
#         """
#         Set values using the _AtIndexer.
#         :param key: a tuple of row index and column name.
#         :param value: a value to set.
#         """
#         # This as no real effect, it is done to check that the key (index, column) exists in the view.
#         self.__at[key] = value
#         # Actually set a value at the (index, column) key.
#         self.__parent.df_data.at[key] = value
#
#
# class _ViewViAtIndexer:
#     """
#     Wrapper around pandas _iAtIndexer object for use in ViewTemporalDataFrames.
#     """
#     def __init__(self, parent: TemporalDataFrame):
#         """
#         :param parent: a parent TemporalDataFrame from a ViewTemporalDataFrames.
#         """
#         self.__parent = parent
#         self.__iat = parent.df_data.iat
#         self.__index = parent.df_data.index
#         self.__columns = parent.df_data.columns
#
#     def __getitem__(self, key: Tuple[int, int]) -> Any:
#         """
#         Get values using the _AtIndexer.
#         :param key: a tuple of row # and column #
#         :return: the value stored at the row # and column #.
#         """
#         return self.__iat[key]
#
#     def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
#         """
#         Set values using the _AtIndexer.
#         :param key: a tuple of row # and column #.
#         :param value: a value to set.
#         """
#         # This as no real effect, it is done to check that the key (index, column) exists in the view.
#         self.__iat[key] = value
#         # Actually set a value at the (index, column) key.
#         row = list(self.__parent.index).index(self.__index[key[0]])
#         col = list(self.__parent.columns).index(self.__columns[key[1]])
#         generalLogger.debug(f"Setting value '{value}' at row '{key[0]}' in view ('{row}' in DataFrame) and col '"
#                             f"{key[1]}' in view ('{col}' in DataFrame).")
#         # increment col by 1 because '__TPID' is masked
#         self.__parent.df_data.iat[row, col+1] = value
