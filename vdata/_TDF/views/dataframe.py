# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Collection, Optional, Union, Tuple, Any, Dict, List, NoReturn, Hashable, IO, Iterable
from typing_extensions import Literal

import vdata
from vdata.NameUtils import PreSlicer, DType
from vdata.utils import repr_array, repr_index, reformat_index, TimePoint
from ..NameUtils import ViewTemporalDataFrame_internal_attributes
from ..._IO import generalLogger
from ..._IO.errors import VValueError, VAttributeError


# ==========================================
# code
class ViewTemporalDataFrame:
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    def __init__(self, parent: 'vdata.TemporalDataFrame', parent_data,
                 tp_slicer: Collection[TimePoint], index_slicer: Collection, column_slicer: Collection):
        """
        :param parent: a parent TemporalDataFrame to view.
        :param parent_data: the parent TemporalDataFrame's data.
        :param tp_slicer: a collection of time points to view.
        :param index_slicer: a pandas Index of rows to view.
        :param column_slicer: a pandas Index of columns to view.
        """
        generalLogger.debug(f"\u23BE ViewTemporalDataFrame '{parent.name}':{id(self)} creation : begin "
                            f"---------------------------------------- ")

        # set attributes on init using object's __setattr__ method to avoid self's __setattr__ which would provoke bugs
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_parent_data', parent_data)
        object.__setattr__(self, 'index', pd.Index(index_slicer))
        object.__setattr__(self, 'columns', column_slicer)

        # remove time points where the index does not match
        object.__setattr__(self, '_tp_slicer', np.array(tp_slicer)[
            [any(self.index.isin(self._parent_data[time_point].index)) for time_point in tp_slicer]])

        generalLogger.debug(f"  1. Refactored time point slicer to : {repr_array(self._tp_slicer)}")

        # remove index elements where time points do not match
        if len(self._tp_slicer):
            valid_indexes = np.concatenate([self._parent_data[time_point].index.values
                                            for time_point in self._tp_slicer])
            index_at_tp_slicer = pd.Index(np.intersect1d(self.index, valid_indexes))

        else:
            index_at_tp_slicer = pd.Index([], dtype=object)

        object.__setattr__(self, 'index', index_at_tp_slicer)

        generalLogger.debug(f"  2. Refactored index slicer to : {repr_array(self.index)}")

        generalLogger.debug(f"\u23BF ViewTemporalDataFrame '{parent.name}':{id(self)} creation : end "
                            f"------------------------------------------ ")

    def __repr__(self):
        """
        Description for this view of a TemporalDataFrame object to print.
        :return: a description of this view of a TemporalDataFrame object
        """
        if self.n_time_points:
            repr_str = f"View of TemporalDataFrame '{self._parent.name}'\n"
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

        else:
            repr_str = f"Empty View of TemporalDataFrame '{self._parent.name}'\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    def one_TP_repr(self, time_point: TimePoint, n: Optional[int] = None, func: Literal['head', 'tail'] = 'head'):
        """
        Representation of a single time point in this TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :param func: the name of the function to use to limit the output ('head' or 'tail')
        :return: a representation of a single time point in this TemporalDataFrame object
        """
        if time_point not in self._tp_slicer:
            raise VValueError(f"TimePoint '{time_point}' is not present in this view.")

        return repr(self._parent_data[time_point].loc[self.index_at(time_point), self.columns].__getattr__(func)(n=n))

    def __getitem__(self, index: Union[PreSlicer,
                                       Tuple[PreSlicer],
                                       Tuple[PreSlicer, PreSlicer],
                                       Tuple[PreSlicer, PreSlicer, PreSlicer]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a sub-view from this view using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index.
            See TemporalDataFrame's '__getitem__' method for more details.
        """
        generalLogger.debug(f"ViewTemporalDataFrame '{self._parent.name}':{id(self)} sub-setting "
                            f"- - - - - - - - - - - - - -")
        generalLogger.debug(f'  Got index : {repr_index(index)}')

        index = reformat_index(index, self.time_points, self.index, self.columns)

        generalLogger.debug(f'  Refactored index to : {repr_index(index)}')

        if not len(index[0]):
            raise VValueError("Time point not found.")

        return ViewTemporalDataFrame(self._parent, self._parent_data, index[0], index[1], index[2])

    def __setitem__(self, index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, Collection[bool]]],
                    df: Union[pd.DataFrame, 'vdata.TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
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
        if attr not in ViewTemporalDataFrame_internal_attributes:
            raise AttributeError

        return object.__getattribute__(self, attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Get columns in the DataFrame (this is done for maintaining the pandas DataFrame behavior).
        :param attr: an attribute's name
        :return: a column with name <attr> from the DataFrame
        """
        if any(np.isin([attr], self.columns)):
            return self._parent.loc[self.index, attr]

        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set value for a regular attribute of for a column in the DataFrame.
        :param attr: an attribute's name
        :param value: a value to be set into the attribute
        """
        if attr in ViewTemporalDataFrame_internal_attributes:
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
        return self.n_index

    # TODO : not like this !
    # def __add__(self, value: Union[int, float]) -> 'ViewTemporalDataFrame':
    #     """
    #     Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
    #     :param value: an int or a float to add to values.
    #     :return: a TemporalDataFrame with new values.
    #     """
    #     index = self.index[0] if len(self.index) == 1 else self.index
    #     columns = self.columns[0] if len(self.columns) == 1 else self.columns
    #
    #     self.parent_data.loc[index, columns] += value
    #
    #     return self

    # def __sub__(self, value: Union[int, float]) -> 'ViewTemporalDataFrame':
    #     """
    #     Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
    #     :param value: an int or a float to subtract to values.
    #     :return: a TemporalDataFrame with new values.
    #     """
    #     time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
    #     time_list = self._df['__TPID'] if time_col is None else None
    #
    #     return TemporalDataFrame(self.to_pandas() - value,
    #                              time_list=time_list,
    #                              time_col=time_col,
    #                              time_points=self.time_points,
    #                              index=self.index,
    #                              name=self.name)
    #
    # def __mul__(self, value: Union[int, float]) -> 'ViewTemporalDataFrame':
    #     """
    #     Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
    #     :param value: an int or a float to multiply all values by.
    #     :return: a TemporalDataFrame with new values.
    #     """
    #     time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
    #     time_list = self._df['__TPID'] if time_col is None else None
    #
    #     return TemporalDataFrame(self.to_pandas() * value,
    #                              time_list=time_list,
    #                              time_col=time_col,
    #                              time_points=self.time_points,
    #                              index=self.index,
    #                              name=self.name)
    #
    # def __truediv__(self, value: Union[int, float]) -> 'ViewTemporalDataFrame':
    #     """
    #     Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
    #     :param value: an int or a float to divide all values by.
    #     :return: a TemporalDataFrame with new values.
    #     """
    #     time_col = self.time_points_column_name if self.time_points_column_name != '__TPID' else None
    #     time_list = self._df['__TPID'] if time_col is None else None
    #
    #     return TemporalDataFrame(self.to_pandas() / value,
    #                              time_list=time_list,
    #                              time_col=time_col,
    #                              time_points=self.time_points,
    #                              index=self.index,
    #                              name=self.name)

    def set(self, df: Union[pd.DataFrame, 'vdata.TemporalDataFrame', 'ViewTemporalDataFrame']) -> None:
        """
        Set values for this ViewTemporalDataFrame.
        Values can be given in a pandas.DataFrame, a TemporalDataFrame or an other ViewTemporalDataFrame.
        To set values, the columns and indexes must match ALL columns and indexes in this ViewTemporalDataFrame.
        :param df: a pandas.DataFrame, TemporalDataFrame or ViewTemporalDataFrame with new values to set.
        """
        assert isinstance(df, (pd.DataFrame, vdata.TemporalDataFrame, ViewTemporalDataFrame)), \
            "Cannot set values from non DataFrame object."
        # This is done to prevent introduction of NaNs
        assert self.n_columns == len(df.columns), "Columns must match."
        assert self.columns.equals(df.columns), "Columns must match."
        assert len(self) == len(df), "Number of rows must match."
        assert self.index.equals(df.index), "Indexes must match."

        if isinstance(df, pd.DataFrame):
            self.parent_data.loc[self.index, self.columns] = df

        else:
            self.parent_data.loc[self.index, self.columns] = df.df_data

    def to_pandas(self, with_time_points: Optional[str] = None) -> Any:
        """
        TODO
        :param with_time_points:
        """
        # index = self.index[0] if len(self.index) == 1 else self.index
        # columns = self.columns[0] if len(self.columns) == 1 else self.columns

        data = pd.DataFrame(columns=self.columns)

        for time_point in self.time_points:
            data = pd.concat((data, self._parent_data[time_point].loc[self.index_at(time_point), self.columns]))

        if with_time_points is not None:
            if with_time_points not in self.columns:
                data[with_time_points] = self._parent.time_points_column.values

            else:
                raise VValueError(f"Column '{with_time_points}' already exists.")

        return data

    @property
    def parent_time_points_col(self) -> str:
        """
        Get parent's _time_points_col.
        :return: parent's _time_points_col
        """
        return getattr(self._parent, '_time_points_col')

    @property
    def time_points(self) -> List[TimePoint]:
        """
        Get the list of time points in this view of a TemporalDataFrame.
        :return: the list of time points in this view of a TemporalDataFrame.
        """
        return self._tp_slicer

    @property
    def n_time_points(self) -> int:
        """
        Get the number of time points.
        :return: the number of time points.
        """
        return len(self.time_points)

    @property
    def time_points_column_name(self) -> Optional[str]:
        """
        Get the name of the column with time points data. Returns None if '__TPID' is used.
        :return: the name of the column with time points data.
        """
        return self._parent.time_points_column_name

    @property
    def time_points_column(self) -> pd.Series:
        """
        Get the time points data for all rows in this TemporalDataFrame.
        :return: the time points data.
        """
        _data = pd.Series([])

        for time_point in self.time_points:
            _data = pd.concat((_data, pd.Series(np.repeat(time_point, self.n_index_at(time_point)))))

        return _data

    def index_at(self, time_point: TimePoint) -> pd.Index:
        """TODO"""
        if time_point not in self._tp_slicer:
            raise VValueError(f"TimePoint '{time_point}' cannot be found in this view.")

        return self._parent_data[time_point].index.intersection(self.index, sort=False)

    @property
    def n_index(self) -> int:
        """
        Get the number of indexes.
        :return: the number of indexes.
        """
        return sum([self.n_index_at(TP) for TP in self.time_points])

    def n_index_at(self, time_point: TimePoint) -> int:
        """
        :return: the length of the index at a given time point.
        """
        return len(self.index_at(time_point))

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
        return self.df_data[self.columns].values

    @property
    def axes(self) -> List[pd.Index]:
        """
        Return a list of the row axis labels.
        :return: a list of the row axis labels.
        """
        return self.df_data[self.columns].axes

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
        return self.df_data[self.columns].size

    @property
    def shape(self) -> Tuple[int, List[int], int]:
        """
        Return a tuple representing the dimensionality of the DataFrame
        (nb_time_points, [n_index_at(time point) for all time points], nb_col)
        :return: a tuple representing the dimensionality of the DataFrame
        """
        return self.n_time_points, [self.n_index_at(TP) for TP in self.time_points], self.n_columns

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
        if not self.n_time_points or not self.n_columns or not self.n_index:
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

    # def items_at(self, time_point: TimePoint) -> Tuple[Optional[Hashable], pd.Series]:
    #     """
    #     Iterate over (column name, Series) pairs.
    #     :return: a tuple with the column name and the content as a Series.
    #     """
    #     return self._parent_data[time_point].items()

    def keys(self) -> List[Optional[Hashable]]:
        """
        Get the ‘info axis’.
        :return: the ‘info axis’.
        """
        if self.n_time_points:
            return list(self._parent_data[self.time_points[0]].keys())

        else:
            return []

    def isin(self, values: Union[Iterable, pd.Series, pd.DataFrame, Dict]) -> 'vdata.TemporalDataFrame':
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

        return vdata.TemporalDataFrame(self.parent_data.isin(values)[self.columns], time_points=time_points,
                                       time_col=time_col)

    def eq(self, other: Any, axis: Literal[0, 1, 'index', 'column'] = 'columns',
           level: Any = None) -> 'vdata.TemporalDataFrame':
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

        return vdata.TemporalDataFrame(self._df.eq(other, axis, level)[self.columns],
                                       time_points=time_points, time_col=time_col)

    # TODO : copy method

    def __mean_min_max_func(self, func: Literal['mean', 'min', 'max'], axis) -> Tuple[Dict, np.ndarray, pd.Index]:
        """
        Compute mean, min or max of the values over the requested axis.
        """
        if axis == 0:
            _data = {'mean': [self._parent_data[tp].loc[self.index_at(tp), col].__getattr__(func)()
                              for tp in self.time_points for col in self.columns]}
            _time_list = np.repeat(self.time_points, self.n_columns)
            _index = pd.Index(np.concatenate([self.columns for _ in range(self.n_time_points)]))

        elif axis == 1:
            _data = {'mean': [self._parent_data[tp].loc[row, self.columns].__getattr__(func)()
                              for tp in self.time_points for row in self.index_at(tp)]}
            _time_list = self.time_points_column
            _index = self.index

        else:
            raise VValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    def mean(self, axis: Literal[0, 1] = 0) -> 'vdata.TemporalDataFrame':
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('mean', axis)

        _name = f"Mean of {self._parent.name}'s view" if self._parent.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def min(self, axis: Literal[0, 1] = 0) -> 'vdata.TemporalDataFrame':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self._parent.name}'s view" if self._parent.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def max(self, axis: Literal[0, 1] = 0) -> 'vdata.TemporalDataFrame':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self._parent.name}'s view" if self._parent.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)
