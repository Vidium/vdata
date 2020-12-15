# coding: utf-8
# Created on 12/9/20 9:17 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
from pandas.core.indexing import _AtIndexer, _iAtIndexer, _LocIndexer, _iLocIndexer
import numpy as np
from typing import Dict, Union, Optional, Collection, Tuple, Any, List, IO, Hashable, Iterable, NoReturn
from typing_extensions import Literal

from .._IO.errors import VValueError, VTypeError, ShapeError
from .._IO.logger import generalLogger
from ..NameUtils import DType, PreSlicer
from ..utils import slice_to_range, isCollection, to_str_list


# ====================================================
# code
def match(tp_list: pd.Series, tp_index: Collection[str]) -> List[bool]:
    """
    TODO
    """
    mask = [False for _ in range(len(tp_list))]

    if len(tp_index):
        for tp_i, tp_obs in enumerate(tp_list):
            if tp_obs == '*':
                mask[tp_i] = True

            else:
                if not isCollection(tp_obs):
                    tp_obs_iter = (tp_obs,)
                else:
                    tp_obs_iter = map(str, tp_obs)

                for one_tp_obs in tp_obs_iter:
                    if one_tp_obs in tp_index:
                        mask[tp_i] = True
                        break
    return mask


class TemporalDataFrame:
    """
    An extension of pandas DataFrames to include a notion of time on the rows.
    An hidden column '__TPID' contains for each row the list of time points this row appears in.
    This class implements a modified sub-setting mechanism to subset on time points and on the regular conditional
    selection.
    """

    _internal_attributes = ['_time_points_col', '_df',
                            'time_points', 'columns', 'index', 'n_time_points', 'n_columns',
                            'dtypes', 'values', 'axes', 'ndim', 'size', 'shape', 'empty',
                            'at', 'iat', 'loc', 'iloc']

    def __init__(self, data: Optional[Union[Dict, pd.DataFrame]] = None,
                 time_points: Optional[Union[Collection, DType, Literal['*']]] = None,
                 time_col: Optional[str] = None,
                 index: Optional[Collection] = None,
                 columns: Optional[Collection] = None,
                 dtype: Optional[DType] = None):
        """
        :param data: data to store as a dataframe
        :param time_points: time points for the dataframe's rows. The value indicates at which time point a given row
            exists in the dataframe.
            It can be :
                - a collection of values of the same length as the number of rows.
                - a single value to set for all rows.

            In any case, the values can be :
                - a single time point (indicating that the row only exists at that given time point)
                - a collection of time points (indicating that the row exists at all those time points)
                - the character '*' (indicating that the row exists at all time points)

        :param time_col: if time points are not given explicitly with the 'time_points' parameter, a column name can be
            given. This column will be used as the time data.
        :param index: indexes for the dataframe's rows
        :param columns: column labels
        :param dtype: data type to force
        """
        self._time_points_col = '__TPID'

        if time_points is not None:
            # time_points = list(map(str, time_points))
            time_points = to_str_list(time_points)

        if columns is not None:
            columns = ['__TPID'] + list(columns)

        # no data given, empty DataFrame is created
        if data is None or (isinstance(data, dict) and not len(data.keys())) or (isinstance(data, pd.DataFrame) and
                                                                                 not len(data.columns)):
            generalLogger.debug('Setting empty TemporalDataFrame.')

            if time_points is None:
                time_points = np.repeat('0', len(index)) if index is not None else []

            if time_col is not None:
                generalLogger.warning("Both 'time_points' and 'time_col' parameters were set, 'time_col' will be "
                                      "ignored.")

            self._df = pd.DataFrame({'__TPID': time_points}, index=index, columns=columns, dtype=dtype)

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

                if '__TPID' in data.keys():
                    raise VValueError("'__TPID' key is reserved and cannot be used in 'data'.")

            else:
                data_len = len(data)
                # data = {col: data[col].values for col in data.columns}

                if '__TPID' in data.columns:
                    raise VValueError("'__TPID' column is reserved and cannot be used in 'data'.")

            # no time points given
            if time_points is None:
                # no column to use as time point : all time points set to 0 by default
                if time_col is None:
                    generalLogger.info(f"Setting all time points to default value '0'.")
                    time_points = ['0' for _ in range(data_len)]

                # a column has been given to use as time point : check that it exists
                elif (isinstance(data, dict) and time_col in data.keys()) or \
                        (isinstance(data, pd.DataFrame) and time_col in data.columns):
                    generalLogger.info(f"Using '{time_col}' as time points data.")
                    time_points = data[time_col]
                    self._time_points_col = time_col

                else:
                    raise VValueError(f"'{time_col}' could not be found in the supplied DataFrame's columns.")

            # time points given, check length is correct
            else:
                if time_col is not None:
                    generalLogger.warning("Both 'time_points' and 'time_col' parameters were set, 'time_col' will be "
                                          "ignored.")

                if len(time_points) != data_len:
                    raise VValueError(f"Supplied time points must be of length {data_len}.")

            # check that all values in time_points are hashable
            hashable_time_points = []
            for tp in time_points:
                try:
                    hash(tp)
                except TypeError:
                    if isinstance(tp, (list, np.ndarray)):
                        hashable_time_points.append(tuple(tp))
                    else:
                        raise VTypeError(f"Un-hashable type '{type(tp)}' cannot be used for time points.")
                else:
                    hashable_time_points.append(tp)

            if isinstance(data, dict):
                data = dict({'__TPID': hashable_time_points}, **data)

            else:
                data.insert(0, "__TPID", hashable_time_points)

            generalLogger.debug('Setting TemporalDataFrame from data.')
            self._df = pd.DataFrame(data, index=index, columns=columns,
                                    dtype=dtype)

        else:
            raise VTypeError(f"Type {type(data)} is not handled for 'data' parameter.")

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
        Get data from the DataFrame using an index with the usual sub-setting mechanics.
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
        # TODO : is this satisfying ? we copy the data to a completely new TDF, this can be long and mem-intensive ...
        # TODO : use views ?
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
            data_for_TP = self._df[match(self._df['__TPID'], index_0_tp)]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self, index_0_tp, data.index, data.columns[1:])

        # Case 2 : sub-setting on columns
        index_0_col = [idx for idx in index[0] if idx in self.columns]
        if len(index_0_col):
            data_for_TP = self._df[index_0_col]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self, self.time_points, data.index, data.columns)

        else:
            data = pd.DataFrame()

            return ViewTemporalDataFrame(self, (), data.index, data.columns[1:])

        # TODO : remove if not needed anymore
        # data_for_TP = self._df[match(self._df['__TPID'], index[0])]
        # index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
        #
        # # get attributes
        # data = data_for_TP[index_conditions]
        # # time_points = data['__TPID'] if self._time_points_col == '__TPID' else None
        # # time_col = self._time_points_col if self._time_points_col != '__TPID' else None
        # # data_index = data.index

        # return TemporalDataFrame(data=data[self._df.columns[1:]], time_points=time_points, time_col=time_col,
        #                          index=data_index)

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    tdf: 'TemporalDataFrame') -> None:
        """
        Set values in the DataFrame with a pandas DataFrame.
        The columns and the number of rows must match.
        :param index: a sub-setting index. (see __getitem__ for more details)
        :param tdf: a TemporalDataFrame with values to set.
        """
        # TODO : does not work because we need views on TDF
        # TODO : what is needed here ?
        # assert isinstance(df, pd.DataFrame), "Cannot set values from non pandas DataFrame object."
        # assert self.n_columns == len(df.columns), "Columns must match."
        # assert self.columns.equals(df.columns), "Columns must match."
        # assert len(self[index]) == len(df), "Number of rows must match."

        # TODO
        self[index] = tdf

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

    @property
    def time_points(self) -> List[str]:
        """
        Get the list of time points in this TemporalDataFrame
        :return: the list of time points in this TemporalDataFrame
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
    def columns(self) -> pd.Index:
        """
        Get the columns of the DataFrame (but mask the reserved __TPID column).
        :return: the column names of the DataFrame
        """
        return self._df.columns[1:]

    @property
    def index(self) -> pd.Index:
        """
        Get the index of the DataFrame.
        :return: the index of the DataFrame
        """
        return self._df.index

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
        return self._df[self.columns].dtypes

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        TODO
        """
        self._df.astype(dtype)

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None,
             memory_usage: Optional[Union[bool, str]] = None, null_counts: Optional[bool] = None) -> None:
        """
        This method prints information about a DataFrame including the index dtype and columns, non-null values and
        memory usage.
        :return: a concise summary of a DataFrame.
        """
        return self._df.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage,
                             null_counts=null_counts)

    @property
    def values(self) -> np.ndarray:
        """
        Return a Numpy representation of the DataFrame.
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
        return self._df.memory_usage(index=index, deep=deep)

    @property
    def empty(self) -> bool:
        """
        Indicator whether DataFrame is empty.
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
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.
        :return: a single value for a row/column label pair.
        """
        return self._df.at

    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.
        :return: a single value for a row/column pair by integer position.
        """
        return self._df.iat

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
        return _VLocIndexer(self._df, self._df.loc, self._time_points_col)

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
        return _ViLocIndexer(self._df.iloc)

    def insert(self, loc, column, value, allow_duplicates=False) -> None:
        """
        Insert column into DataFrame at specified location.
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
        Whether each element in the DataFrame is contained in values.
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


class ViewTemporalDataFrame:
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    _internal_attributes = ['parent', '_tp_slicer']

    def __init__(self, parent: TemporalDataFrame, time_point_slicer, index_slicer, column_slicer):
        """
        :param parent:
        :param time_point_slicer:
        :param index_slicer:
        :param column_slicer:
        """
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_tp_slicer', time_point_slicer)
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
        mask = np.array(match(self._parent._df['__TPID'], time_point)) & np.array(self.index_bool)
        return repr(self._parent._df.loc[mask, self.columns].__getattr__(func)(n=n))

    def __getitem__(self, index):
        """
        TODO
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
            mask = np.array(match(self._parent._df['__TPID'], index_0_tp)) & np.array(self.index_bool)

            data_for_TP = self._parent._df[mask]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self._parent, index_0_tp, data.index, data.columns[1:])

        # Case 2 : sub-setting on columns
        index_0_col = [idx for idx in index[0] if idx in self.columns]
        if len(index_0_col):
            data_for_TP = self._parent._df.loc[self.index_bool, index_0_col]
            index_conditions = index[1][data_for_TP.index] if not isinstance(index[1], slice) else index[1]
            data = data_for_TP[index_conditions]

            return ViewTemporalDataFrame(self._parent, self.time_points, data.index, data.columns)

        else:
            data = pd.DataFrame()

            return ViewTemporalDataFrame(self._parent, (), data.index, data.columns[1:])

    def __setitem__(self):
        """
        TODO
        """
        pass

    def __getattribute__(self, attr: str) -> Any:
        """
        Get attribute from this TemporalDataFrame in obj.attr fashion.
        This is called before __getattr__.
        :param attr: an attribute's name to get.
        :return: self.attr
        """
        if attr not in TemporalDataFrame._internal_attributes \
                and attr not in ViewTemporalDataFrame._internal_attributes:
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
        if attr in TemporalDataFrame._internal_attributes \
                and attr not in ViewTemporalDataFrame._internal_attributes:
            self._parent.__setattr__(attr, value)

        elif attr in self.columns:
            self._parent._df.loc[self.index, attr] = value

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __len__(self):
        """
        Returns the length of info axis.
        :return: the length of info axis.
        """
        return len(self.index)

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
        return self._parent._df[self.columns].dtypes

    def astype(self, dtype: Union[DType, Dict[str, DType]]) -> None:
        """
        TODO
        """
        self._parent._df[self.columns].astype(dtype)

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
        return self._parent._df[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self._parent._df[self.columns].axes

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
        return self._parent._df[self.columns].size

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
            mask = np.array(match(self._parent._df['__TPID'], TP)) & np.array(self.index_bool)
            if not self._parent._df[mask].empty:
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

    # TODO : test at, iat, loc and iloc for value setting
    @property
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.
        :return: a single value for a row/column label pair.
        """
        return self._parent._df[self.index_bool].at

    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.
        :return: a single value for a row/column pair by integer position.
        """
        return self._parent._df[self.index_bool].iat

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
        return _VLocIndexer(self._parent._df[self.index_bool], self._parent._df[self.index_bool].loc,
                            self._parent._time_points_col)

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
        return _ViLocIndexer(self._parent._df[self.index_bool].iloc)

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
        return list(self._parent._df[self.index_bool].items())[1:]

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        keys = list(self._parent._df[self.index_bool].keys())

        if '__TPID' in keys:
            keys.remove('__TPID')

        return keys

    # TODO : think about isin and eq --> return view or TDF ?
    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'TemporalDataFrame':
        """
        Whether each element in the DataFrame is contained in values.
        :return: whether each element in the DataFrame is contained in values.
        """
        if self._time_points_col == '__TPID':
            time_points = self._parent._df[self.index_bool]['__TPID'].tolist()
            time_col = None

        else:
            time_points = self._parent._df[self.index_bool][self._time_points_col].tolist()
            time_col = self._parent._time_points_col

        return TemporalDataFrame(self._parent._df.isin(values)[self.columns], time_points=time_points,
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


# TODO : check this works : it returns new TDF and not views on TDF
class _VLocIndexer:
    """
    A simple wrapper around the pandas _LocIndexer object.
    """

    def __init__(self, df: pd.DataFrame, loc: _LocIndexer, time_col: str):
        """
        :param df: a pandas DataFrame to subset (from a TemporalDataFrame)
        :param loc: a pandas _LocIndexer.
        :param time_col: the name of the column which contains the time points
        """
        self.df = df
        self.loc = loc
        self.time_col = time_col

    def __getitem__(self, index: Any) -> TemporalDataFrame:
        """
        Get rows and columns from the loc.
        :param index: an index for getting rows and columns.
        :return: a TemporalDataFrame built from the rows and columns accessed from the loc.
        """
        new_df = pd.DataFrame(self.loc[index])

        if self.time_col in new_df.columns:
            time_points = None
            time_col = self.time_col

        else:
            time_points = self.df.loc[new_df.index, '__TPID'].tolist()
            time_col = None

        return TemporalDataFrame(new_df, time_points=time_points, time_col=time_col)

    # def __setitem__(self, index, value) -> None:
    #     """
    #     TODO
    #     """
    #     pass


class _ViLocIndexer:
    """
    A simple wrapper around the pandas _iLocIndexer object.
    """

    def __init__(self, iloc: _iLocIndexer):
        """
        :param iloc: a pandas _iLocIndexer.
        """
        self.iloc = iloc

    def __getitem__(self, index: Any) -> Any:
        """
        Get rows and columns from the iloc.
        :param index: an index for getting rows and columns.
        :return: a TemporalDataFrame built from the rows and columns accessed from the loc.
        """
        value = self.iloc[index]

        if isinstance(value, pd.Series):
            return value[value.keys()[1:]]

        return value

    # def __setitem__(self, index: Any, value: Any) -> None:
    #     """
    #     TODO
    #     """
    #     self.iloc[index] = value
