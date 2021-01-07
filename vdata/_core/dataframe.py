# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Collection, Tuple, Any, List, IO, Hashable, Iterable, NoReturn
from typing_extensions import Literal

from .._IO.errors import VValueError, VTypeError, ShapeError, VAttributeError
from .._IO.logger import generalLogger
from ..NameUtils import DType, PreSlicer
from ..utils import slice_to_range, isCollection, to_str_list, to_list


# ====================================================
# code
def match(tp_list: pd.Series, tp_index: Collection[str]) -> List[bool]:
    """
    Find where in the tp_list the values in tp_index are present. This function parses the tp_list to understand the
    '*' character (meaning the all values in tp_index match) and tuples of time points.
    :param tp_list: the time points columns in a TemporalDataFrame.
    :param tp_index: a collection of target time points to match in tp_list.
    :return: a list of booleans of the same length as tp_list, where True indicates that a value in tp_list matched
        a value in tp_index.
    """
    mask = [False for _ in range(len(tp_list))]

    if len(tp_index):
        for tp_i, tp_obs in enumerate(tp_list):
            if tp_obs == '*':
                mask[tp_i] = True

            else:
                if not isCollection(tp_obs):
                    tp_obs_iter: Iterable = to_str_list(tp_obs)
                else:
                    tp_obs_iter = map(str, tp_obs)

                for one_tp_obs in tp_obs_iter:
                    if one_tp_obs in tp_index:
                        mask[tp_i] = True
                        break
    return mask


class TemporalDataFrame:
    """
    An extension to pandas DataFrames to include a notion of time on the rows.
    An hidden column '__TPID' contains for each row the list of time points this row appears in.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """

    _internal_attributes = ['_time_points_col', '_df', '_time_points', 'TP_from_DF',
                            'time_points', 'columns', 'index', 'n_time_points', 'n_columns',
                            'dtypes', 'values', 'axes', 'ndim', 'size', 'shape', 'empty',
                            'at', 'iat', 'loc', 'iloc']

    _reserved_keys = ['__TPID', 'df_data']

    def __init__(self, data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_list: Optional[Union[Collection, DType, Literal['*']]] = None,
                 time_col: Optional[str] = None,
                 time_points: Optional[Collection[str]] = None,
                 index: Optional[Collection] = None,
                 columns: Optional[Collection] = None,
                 dtype: Optional[DType] = None):
        """
        :param data: data to store as a dataframe
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
        :param index: indexes for the dataframe's rows
        :param columns: column labels
        :param dtype: data type to force
        """
        generalLogger.debug(u'\u23BE TemporalDataFrame creation : begin ---------------------------------------- ')
        self._time_points_col = '__TPID'
        self._time_points = sorted(to_str_list(time_points)) if time_points is not None else None
        if self._time_points is not None:
            generalLogger.debug(f"User has defined time points '{self._time_points}'.")

        time_list = to_str_list(time_list) if time_list is not None else None

        if columns is not None:
            columns = ['__TPID'] + list(columns)

        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            if time_list is None:
                time_list = np.repeat('0', len(index)) if index is not None else []

            if time_col is not None:
                generalLogger.warning("Both 'time_list' and 'time_col' parameters were set, 'time_col' will be "
                                      "ignored.")

            self._df = pd.DataFrame({'__TPID': time_list}, index=index, columns=columns, dtype=dtype)

        # data given
        elif isinstance(data, (dict, pd.DataFrame)):
            # if data is a dict, check that the dict can be converted to a DataFrame
            if isinstance(data, dict):
                # get number of rows in data
                data_len = 1
                values = data.values() if isinstance(data, dict) else data.values
                for value in values:
                    value_len = len(value) if hasattr(value, '__len__') else 1

                    if value_len != data_len and data_len != 1 and value_len != 1:
                        raise ShapeError("All items in 'data' must have the same length "
                                         "(or be a unique value to set for all rows).")

                    if data_len == 1:
                        data_len = value_len

                generalLogger.debug(f"Found data in a dictionary with {data_len} rows.")

                for key in TemporalDataFrame._reserved_keys:
                    if key in data.keys():
                        raise VValueError(f"'{key}' key is reserved and cannot be used in 'data'.")

            else:
                data_len = len(data)

                generalLogger.debug(f"Found data in a DataFrame with {data_len} rows.")

                for key in TemporalDataFrame._reserved_keys:
                    if key in data.columns:
                        raise VValueError(f"'{key}' column is reserved and cannot be used in 'data'.")

            # no time points given
            if time_list is None:
                # no column to use as time point : all time points set to 0 by default
                if time_col is None:
                    default_TP = self._time_points[0] if self._time_points is not None else '0'
                    generalLogger.info(f"Setting all time points to default value '{default_TP}'.")
                    time_list = [default_TP for _ in range(data_len)]

                # a column has been given to use as time point : check that it exists
                elif (isinstance(data, dict) and time_col in data.keys()) or \
                        (isinstance(data, pd.DataFrame) and time_col in data.columns):
                    generalLogger.info(f"Using '{time_col}' as time points data.")
                    time_list = data[time_col]
                    self._time_points_col = time_col

                else:
                    raise VValueError(f"'{time_col}' could not be found in the supplied DataFrame's columns.")

            # time points given, check length is correct
            else:
                if time_col is not None:
                    generalLogger.warning("Both 'time_list' and 'time_col' parameters were set, 'time_col' will be "
                                          "ignored.")

                if len(time_list) != data_len:
                    raise VValueError(f"Supplied time points must be of length {data_len}.")

            # check that all values in time_list are hashable
            hashable_time_list = []
            for tp in time_list:
                try:
                    hash(tp)
                except TypeError:
                    if isinstance(tp, (list, np.ndarray)):
                        hashable_time_list.append(tuple(tp))
                    else:
                        raise VTypeError(f"Un-hashable type '{type(tp)}' cannot be used for time points.")
                else:
                    hashable_time_list.append(tp)

            if isinstance(data, dict):
                data = dict({'__TPID': hashable_time_list}, **data)

            else:
                data.insert(0, "__TPID", hashable_time_list)

            generalLogger.debug("Storing data in TemporalDataFrame.")
            self._df = pd.DataFrame(data, dtype=dtype)
            if index is not None:
                self._df.index = index
            if columns is not None:
                self._df.columns = columns

            if self._time_points is not None:
                undesired_time_points = np.unique(self._df[~self._df[self._time_points_col].isin(self._time_points + [
                    '*'])][self._time_points_col])
                if len(undesired_time_points):
                    generalLogger.warning(f"Time points {undesired_time_points} were found in 'data' but were not "
                                          f"specified in 'time_points'. They will be deleted from the DataFrame, "
                                          f"is this what you intended ?")
                    # remove undesired time points from the data
                    self._df = self._df[self._df[self._time_points_col].isin(self._time_points + ['*'])]
                    generalLogger.debug(f"New data has {len(self._df)} rows.")

        else:
            raise VTypeError(f"Type {type(data)} is not handled for 'data' parameter.")

        # get list of time points that can be found in the DataFrame
        self.TP_from_DF = self.__get_time_points()
        generalLogger.debug(f"Found time points {self.TP_from_DF} from stored DataFrame.")

        generalLogger.debug(u'\u23BF TemporalDataFrame creation : end ------------------------------------------ ')

    def __repr__(self) -> str:
        """
        Description for this TemporalDataFrame object to print.
        :return: a description of this TemporalDataFrame object
        """
        if self.n_time_points:
            repr_str = ""
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self[TP].one_TP_repr(TP)}\n"

        else:
            repr_str = f"Empty TemporalDataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]]) -> \
            'ViewTemporalDataFrame':
        """
        Get a view from the DataFrame using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index. It can be a single index or a 2-tuple of indexes.
            An index can be a string, an int, a float, a sequence of those, a range, a slice or an ellipsis ('...').
            Single indexes are converted to a 2-tuple :
                * single index --> (index, :)

            The first element in the 2-tuple is the list of time points to select, the second element is a
            collection of bool that can be obtained from conditions on the TemporalDataFrame as it is done with
            pandas DataFrames.

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
        generalLogger.debug('TemporalDataFrame sub-setting - - - - - - - - - - - - - - ')
        generalLogger.debug(f'  Got index : {index}.')

        if not isinstance(index, tuple):
            index = [index, slice(None, None, None)]
        elif isinstance(index, tuple) and len(index) == 1:
            index = [index[0], slice(None, None, None)]
        else:
            index = [index[0], index[1]]

        generalLogger.debug(f'  Refactored index to : {index}.')

        # check first index (subset on time points)
        if isinstance(index[0], type(Ellipsis)) or isinstance(index[0], slice) and index[0] == slice(None, None, None):
            index[0] = self.time_points

        elif isinstance(index[0], slice):
            try:
                list(map(int, self.time_points))
            except ValueError:
                raise VTypeError("Cannot slice on time points since time points are not ints.")

            max_value = int(max(self.time_points))

            index[0] = list(map(str, slice_to_range(index[0], max_value)))

        else:
            index[0] = to_str_list(index[0])

        generalLogger.debug(f'  Corrected index to : {index}.')

        # Case 1 : sub-setting on time points
        index_0_tp = [idx for idx in index[0] if idx in self.time_points]
        if len(index_0_tp):
            generalLogger.debug('  Sub-set on time points.')

            data_for_TP = self._df[match(self._df['__TPID'], index_0_tp)]

            index_conditions = [index[1][i] for i in np.where(self._df.index.isin(data_for_TP.index))[0]] if not \
                isinstance(index[1], slice) else index[1]

            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self, index_0_tp, data.index, data.columns[1:])

        # Case 2 : sub-setting on columns
        index_0_col = [idx for idx in index[0] if idx in self.columns]
        if len(index_0_col):
            generalLogger.debug('  Sub-set on columns.')

            data_for_TP = self._df[index_0_col]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self, self.time_points, data.index, data.columns)

        else:
            raise VValueError('Sub-setting index was not understood. If you meant to sub-set on rows, '
                              'use TDF[:, <List of booleans>]')

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
        if attr not in TemporalDataFrame._internal_attributes:
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
        if attr in TemporalDataFrame._internal_attributes:
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
        return len(self._df)

    def __get_time_points(self) -> List[str]:
        """
        Get the list of time points in this TemporalDataFrame as defined in <_time_points_col>.
        :return: the list of time points in this TemporalDataFrame as defined in <_time_points_col>.
        """
        all_values = to_str_list(set(self._df[self._time_points_col].values))

        unique_values = set()
        for value in all_values:
            if isCollection(value):
                unique_values.union(set(value))

            else:
                unique_values.add(value)

        return sorted(map(str, unique_values - {'*'}))

    @property
    def df_data(self) -> pd.DataFrame:
        """
        Get the raw pandas.DataFrame stored in the TemporalDataFrame.
        :return: the raw pandas.DataFrame stored in the TemporalDataFrame.
        """
        return self._df

    @property
    def time_points(self) -> List[str]:
        """
        Get the list of time points in this TemporalDataFrame.
        :return: the list of time points in this TemporalDataFrame.
        """
        if self._time_points is not None:
            return self._time_points

        elif len(self.TP_from_DF):
            return self.TP_from_DF

        else:
            return [] if self._df.empty else ['0']

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this TemporalDataFrame (but mask the reserved __TPID column).
        :return: the column names of this TemporalDataFrame
        """
        return self._df.columns[1:]

    @columns.setter
    def columns(self, values: pd.Index) -> None:
        """
        Set the columns of this TemporalDataFrame (except for __TPID).
        :param values: the new column names for this TemporalDataFrame.
        """
        self._df.columns[1:] = values

    @property
    def index(self) -> pd.Index:
        """
        Get the index of this TemporalDataFrame.
        :return: the index of this TemporalDataFrame
        """
        return self._df.index

    @index.setter
    def index(self, values: pd.Index) -> None:
        """
        Set the index of this TemporalDataFrame.
        :param values: the new index to set.
        """
        self._df.index = values

    @property
    def n_time_points(self) -> int:
        """
        Get the number of distinct time points in this TemporalDataFrame.
        :return: the number of time points.
        """
        return len(self.time_points)

    def len_index(self, time_point: str) -> int:
        """
        Get the length of the index at a given time point.
        :param time_point: a time points in this TemporalDataFrame.
        :return: the length of the index at a given time point.
        """
        return len(self[time_point].index)

    @property
    def n_columns(self) -> int:
        """
        Get the number of columns in this TemporalDataFrame
        :return: the number of columns
        """
        return len(self.columns)

    @property
    def dtypes(self) -> None:
        """
        Return the dtypes in this TemporalDataFrame.
        :return: the dtypes in this TemporalDataFrame.
        """
        return self._df[self.columns].dtypes

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        Cast this TemporalDataFrame to a specified data type.
        :param dtype: a data type.
        """
        self._df.astype(dtype)

    def asColType(self, col_name: str, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        Cast a specific column in this TemporalDataFrame to a specified data type.
        :param col_name: a column name.
        :param dtype: a data type.
        """
        self._df[col_name].astype(dtype)

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a TemporalDataFrame including the index dtype and columns,
        non-null values and memory usage.
        :return: a concise summary of a DataFrame.
        """
        return self._df.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                             null_counts=null_counts)

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
        (nb_time_points, [len_index for all time points], nb_col).
        :return: a tuple representing the dimensionality of this TemporalDataFrame
        """
        return self.n_time_points, [self.len_index(TP) for TP in self.time_points], self.n_columns

    def memory_usage(self, index: bool = True, deep: bool = False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        return self._df.memory_usage(index=index, deep=deep)

    @property
    def empty(self) -> bool:
        """
        Indicator whether this TemporalDataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        return True if not self.n_time_points else all([self[TP].empty for TP in self.time_points])

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
        self._df.insert(loc, column, value, allow_duplicates)

    def items(self) -> List[Tuple[Optional[Hashable], pd.Series]]:
        """
        Iterate over (column name, Series) pairs.
        :return: a tuple with the column name and the content as a Series.
        """

        return list(self._df.items())[1:]

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        keys = list(self._df.keys())

        if '__TPID' in keys:
            keys.remove('__TPID')

        return keys

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


class ViewTemporalDataFrame:
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    _internal_attributes = getattr(TemporalDataFrame, '_internal_attributes') + ['parent']

    def __init__(self, parent: TemporalDataFrame, tp_slicer: Collection[str], index_slicer: pd.Index,
                 column_slicer: pd.Index):
        """
        :param parent: a parent TemporalDataFrame to view.
        :param tp_slicer: a collection of time points to view.
        :param index_slicer: a pandas Index of rows to view.
        :param column_slicer: a pandas Index of columns to view.
        """
        # set attributes on init using object's __setattr__ method to avoid self's __setattr__ which would provoke bugs
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_tp_slicer', tp_slicer)
        object.__setattr__(self, 'index', index_slicer)
        object.__setattr__(self, 'columns', column_slicer)

    def __repr__(self):
        """
        Description for this view of a TemporalDataFrame object to print.
        :return: a description of this view of a TemporalDataFrame object
        """
        if self.n_time_points:
            repr_str = ""
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n"

        else:
            repr_str = f"Empty View of a TemporalDataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    def one_TP_repr(self, time_point: str, n: Optional[int] = None, func: Literal['head', 'tail'] = 'head'):
        """
        Representation of a single time point in this TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :param func: the name of the function to use to limit the output ('head' or 'tail')
        :return: a representation of a single time point in this TemporalDataFrame object
        """
        m = match(self.parent_data['__TPID'], time_point)
        if len(m):
            mask = np.array(match(self.parent_data['__TPID'], time_point)) & np.array(self.index_bool)
            return repr(self.parent_data.loc[mask, self.columns].__getattr__(func)(n=n))

        else:
            repr_str = f"Empty DataFrame\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"
            return repr_str

    def __getitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a sub-view from this view using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index.
            See TemporalDataFrame's '__getitem__' method for more details.
        """
        if not isinstance(index, tuple):
            index = [index, slice(None, None, None)]
        elif isinstance(index, tuple) and len(index) == 1:
            index = [index[0], slice(None, None, None)]
        else:
            index = [index[0], index[1]]

        # check first index (subset on time points)
        if isinstance(index[0], type(Ellipsis)) or index[0] == slice(None, None, None):
            index[0] = self.time_points

        elif isinstance(index[0], slice):
            try:
                list(map(int, self.time_points))
            except ValueError:
                raise VTypeError("Cannot slice on time points since time points are not ints.")

            max_value = int(max(self.time_points))

            index[0] = list(map(str, slice_to_range(index[0], max_value)))

        else:
            index[0] = to_str_list(index[0])

        # Case 1 : sub-setting on time points
        index_0_tp = [idx for idx in index[0] if idx in self.time_points]
        if len(index_0_tp):
            mask = np.array(match(self.parent_data['__TPID'], index_0_tp)) & np.array(self.index_bool)

            data_for_TP = self.parent_data[mask]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self._parent, index_0_tp, data.index, data.columns[1:])

        # Case 2 : sub-setting on columns
        index_0_col = [idx for idx in index[0] if idx in self.columns]
        if len(index_0_col):
            data_for_TP = self.parent_data.loc[self.index_bool, index_0_col]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self._parent, self.time_points, data.index, data.columns)

        else:
            data = pd.DataFrame()

            return ViewTemporalDataFrame(self._parent, (), data.index, data.columns[1:])

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    df: Union[pd.DataFrame, 'TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        """
        Set values in the parent TemporalDataFrame from this view with a DataFrame.
        The columns and the rows must match.
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
        if attr not in ViewTemporalDataFrame._internal_attributes:
            raise AttributeError

        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :param attr: an attribute's name
        :return: a column with name <attr> from the DataFrame
        """
        if attr in self.columns:
            return self._parent.loc[self.index, attr]

        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set value for a regular attribute of for a column in the DataFrame.
        :param attr: an attribute's name
        :param value: a value to be set into the attribute
        """
        if attr in ViewTemporalDataFrame._internal_attributes:
            self._parent.__setattr__(attr, value)

        elif attr in self.columns:
            self.parent_data.loc[self.index, attr] = value

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __len__(self):
        """
        Returns the length of info axis.
        :return: the length of info axis.
        """
        return len(self.index)

    def set(self, df: Union[pd.DataFrame, TemporalDataFrame, 'ViewTemporalDataFrame']) -> None:
        """
        Set values for this ViewTemporalDataFrame.
        Values can be given in a pandas.DataFrame, a TemporalDataFrame or an other ViewTemporalDataFrame.
        To set values, the columns and indexes must match ALL columns and indexes in this ViewTemporalDataFrame.
        :param df: a pandas.DataFrame, TemporalDataFrame or ViewTemporalDataFrame with new values to set.
        """
        assert isinstance(df, (pd.DataFrame, TemporalDataFrame, ViewTemporalDataFrame)), "Cannot set values from non " \
                                                                                         "DataFrame object."
        # This is done to prevent introduction of NaNs
        assert self.n_columns == len(df.columns), "Columns must match."
        assert self.columns.equals(df.columns), "Columns must match."
        assert len(self) == len(df), "Number of rows must match."
        assert self.index.equals(df.index), "Indexes must match."

        if isinstance(df, pd.DataFrame):
            self.parent_data.loc[self.index, self.columns] = df

        else:
            self.parent_data.loc[self.index, self.columns] = df.df_data

    @property
    def parent_data(self) -> pd.DataFrame:
        """
        Get parent's _df.
        :return: parent's _df.
        """
        return getattr(self._parent, '_df')

    @property
    def df_data(self) -> pd.DataFrame:
        """
        Get a view on the parent TemporalDataFrame's raw pandas.DataFrame.
        :return: a view on the parent TemporalDataFrame's raw pandas.DataFrame.
        """
        return self.parent_data.loc[self.index, self.columns]

    @property
    def parent_time_points_col(self) -> str:
        """
        Get parent's _time_points_col.
        :return: parent's _time_points_col
        """
        return getattr(self._parent, '_time_points_col')

    @property
    def index_bool(self) -> List[bool]:
        """
        Returns a list of booleans indicating whether the parental DataFrame's indexes are present in this view
        :return: a list of booleans indicating whether the parental DataFrame's indexes are present in this view
        """
        return [idx in self.index for idx in self._parent.index]

    @property
    def time_points(self) -> List[str]:
        """
        Get the list of time points in this view of a TemporalDataFrame
        :return: the list of time points in this view of a TemporalDataFrame
        """
        return self._tp_slicer

    @property
    def n_time_points(self) -> int:
        """
        :return: the number of time points
        """
        return len(self.time_points)

    def len_index(self, time_point: str) -> int:
        """
        :return: the length of the index at a given time point
        """
        return len(self[time_point].index)

    @property
    def n_columns(self) -> int:
        """
        :return: the number of columns
        """
        return len(self.columns)

    @property
    def dtypes(self) -> None:
        """
        Return the dtypes in the DataFrame.
        :return: the dtypes in the DataFrame.
        """
        return self.parent_data[self.columns].dtypes

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> NoReturn:
        """
        Reference to TemporalDataFrame's astype method. This cannot be done in a view.
        """
        raise VAttributeError('Cannot set data type from a view of a TemporalDataFrame.')

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a DataFrame including the index dtype and columns, non-null values and
        memory usage.
        :return: a concise summary of a DataFrame.
        """
        return self._parent.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                                 null_counts=null_counts)

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame.
        :return: a Numpy representation of the DataFrame.
        """
        return self.parent_data[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self.parent_data[self.columns].axes

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
        return self.parent_data[self.columns].size

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of the DataFrame
        (nb_time_points, [len_index for all time points], nb_col)
        :return: a tuple representing the dimensionality of the DataFrame
        """
        return self.n_time_points, [self.len_index(TP) for TP in self.time_points], self.n_columns

    def memory_usage(self, index: bool = True, deep: bool = False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        return self._parent.memory_usage(index=index, deep=deep)

    @property
    def empty(self) -> bool:
        """
        Indicator whether DataFrame is empty.
        :return: True if this TemporalDataFrame is empty.
        """
        if not self.n_time_points:
            return True

        for TP in self.time_points:
            mask = np.array(match(self.parent_data['__TPID'], TP)) & np.array(self.index_bool)
            if not self.parent_data[mask].empty:
                return False

        return True

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

    # @property
    # def at(self) -> '_ViewVAtIndexer':
    #     """
    #     Access a single value for a row/column label pair.
    #     :return: a single value for a row/column label pair.
    #     """
    #     return _ViewVAtIndexer(self._parent)
    #
    # @property
    # def iat(self) -> '_ViewViAtIndexer':
    #     """
    #     Access a single value for a row/column pair by integer position.
    #     :return: a single value for a row/column pair by integer position.
    #     """
    #     return _ViewViAtIndexer(self._parent)

    # TODO : test loc and iloc for value setting
    # @property
    # def loc(self) -> '_VLocIndexer':
    #     """
    #     Access a group of rows and columns by label(s) or a boolean array.
    #
    #     Allowed inputs are:
    #         - A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an
    #         integer position along the index).
    #         - A list or array of labels, e.g. ['a', 'b', 'c'].
    #         - A slice object with labels, e.g. 'a':'f'.
    #         - A boolean array of the same length as the axis being sliced, e.g. [True, False, True].
    #         - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
    #         for indexing (one of the above)
    #
    #     :return: a group of rows and columns
    #     """
    #     return _VLocIndexer(self.parent_data[self.index_bool], self.parent_data[self.index_bool].loc,
    #                         self.parent_time_points_col)

    # @property
    # def iloc(self) -> '_ViLocIndexer':
    #     """
    #     Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).
    #
    #     Allowed inputs are:
    #         - An integer, e.g. 5.
    #         - A list or array of integers, e.g. [4, 3, 0].
    #         - A slice object with ints, e.g. 1:7.
    #         - A boolean array.
    #         - A callable function with one argument (the calling Series or DataFrame) and that returns valid output
    #         for indexing (one of the above). This is useful in method chains, when you don’t have a reference to the
    #         calling object, but would like to base your selection on some value.
    #
    #     :return: a group of rows and columns
    #     """
    #     return _ViLocIndexer(self.parent_data[self.index_bool].iloc)

    def insert(self, *args, **kwargs) -> NoReturn:
        """
        TODO
        """
        raise VValueError("Cannot insert a column from a view.")

    def items(self) -> List[Tuple[Optional[Hashable], pd.Series]]:
        """
        Iterate over (column name, Series) pairs.
        :return: a tuple with the column name and the content as a Series.
        """
        return list(self.parent_data[self.index_bool].items())[1:]

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        keys = list(self.parent_data[self.index_bool].keys())

        if '__TPID' in keys:
            keys.remove('__TPID')

        return keys

    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'TemporalDataFrame':
        """
        Whether each element in the DataFrame is contained in values.
        :return: whether each element in the DataFrame is contained in values.
        """
        if self._time_points_col == '__TPID':
            time_points = self.parent_data[self.index_bool]['__TPID'].tolist()
            time_col = None

        else:
            time_points = self.parent_data[self.index_bool][self._time_points_col].tolist()
            time_col = self.parent_time_points_col

        return TemporalDataFrame(self.parent_data.isin(values)[self.columns], time_points=time_points,
                                 time_col=time_col)

    def eq(self, other: Any, axis: Literal[0, 1, 'index', 'column'] = 'columns',
           level: Any = None) -> 'TemporalDataFrame':
        """
        Get Equal to of dataframe and other, element-wise (binary operator eq).
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

    # TODO : copy method


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

                start = key[1].start+1 if key[1].start is not None else 1
                stop = key[1].stop+1 if key[1].stop is not None else len(self.__parent.columns)+1
                step = key[1].step if key[1].step is not None else 1

                new_key = [0] + list(range(start, stop, step))

                key = (key[0], new_key)

            elif isinstance(key[1], int):
                generalLogger.debug(f'Second item is an int : update it.')
                if isinstance(key[0], (list, np.ndarray, slice)):
                    key = (key[0], [0, key[1]+1])

                else:
                    key = (key[0], key[1] + 1)

            elif isinstance(key[1], (list, np.ndarray)) and len(key[1]):
                if isinstance(key[1][0], bool):
                    generalLogger.debug(f'Second item is an array of bool : update it.')
                    key = (key[0], [True] + key[1])

                elif isinstance(key[1][0], int):
                    generalLogger.debug(f'Second item is an array of int : update it.')
                    key = (key[0], [0] + [v+1 for v in key[1]])

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
        result = self.__getiLoc(key)

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
        return self.__iat[key[0], key[1]+1]

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        # increment key[1] by 1 because '__TPID' is masked
        # This as no real effect, it is done to check that the key (index, column) exists in the view.
        self.__iat[key[0], key[1]+1] = value
        # Actually set a value at the (index, column) key.
        generalLogger.debug(f"Setting value '{value}' at row '{key[0]}' and col '{key[1]+1}'")
        self.__parent.df_data.iat[key[0], key[1]+1] = value


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
