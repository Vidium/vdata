# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from typing import Collection, Optional, Union, Tuple, Any, Dict, List, NoReturn, IO
from typing_extensions import Literal

import vdata
from vdata.NameUtils import PreSlicer, DType, DataFrame
from vdata.utils import repr_array, repr_index, reformat_index, TimePoint, isCollection
from ..NameUtils import ViewTemporalDataFrame_internal_attributes
from .. import dataframe
from .. import base
from .. import copy
from ..indexers import _VAtIndexer, _ViAtIndexer, _VLocIndexer, _ViLocIndexer
from ..._IO import generalLogger
from ..._IO.errors import VValueError, VAttributeError, VTypeError


# ==========================================
# code
class ViewTemporalDataFrame(base.BaseTemporalDataFrame):
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    __base_repr_str = 'View of TemporalDataFrame'

    def __init__(self, parent: 'vdata.TemporalDataFrame', parent_data: Dict['vdata.TimePoint', DataFrame],
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
        object.__setattr__(self, '_index', pd.Index(index_slicer))
        object.__setattr__(self, '_columns', pd.Index(column_slicer))

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

        object.__setattr__(self, '_index', index_at_tp_slicer)

        generalLogger.debug(f"  2. Refactored index slicer to : {repr_array(self.index)}")

        generalLogger.debug(f"\u23BF ViewTemporalDataFrame '{parent.name}':{id(self)} creation : end "
                            f"------------------------------------------ ")

    def __repr__(self):
        """
        Description for this view of a TemporalDataFrame object to print.
        :return: a description of this view of a TemporalDataFrame object.
        """
        if self.n_time_points:
            repr_str = f"{ViewTemporalDataFrame.__base_repr_str} '{self.name}'\n"
            for TP in self.time_points:
                repr_str += f"\033[4mTime point : {TP}\033[0m\n"
                repr_str += f"{self.one_TP_repr(TP)}\n\n"

        else:
            repr_str = f"Empty {ViewTemporalDataFrame.__base_repr_str} '{self.name}'\n" \
                       f"Columns: {[col for col in self.columns]}\n" \
                       f"Index: {[idx for idx in self.index]}"

        return repr_str

    def one_TP_repr(self, time_point: TimePoint, n: Optional[int] = None, func: Literal['head', 'tail'] = 'head'):
        """
        Representation of a single time point in this view of a TemporalDataFrame to print.
        :param time_point: the time point to represent.
        :param n: the number of rows to print. Defaults to all.
        :param func: the name of the function to use to limit the output ('head' or 'tail').
        :return: a representation of a single time point in this view of a TemporalDataFrame object.
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
        generalLogger.debug(f"ViewTemporalDataFrame '{self.name}':{id(self)} sub-setting "
                            f"- - - - - - - - - - - - - -")
        generalLogger.debug(f'  Got index : {repr_index(index)}')

        index = reformat_index(index, self.time_points, self.index, self.columns)

        generalLogger.debug(f'  Refactored index to : {repr_index(index)}')

        if not len(index[0]):
            raise VValueError("Time point not found.")

        return ViewTemporalDataFrame(self._parent, self._parent_data, index[0], index[1], index[2])

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
            self._parent_data.loc[self.index, attr] = value

        else:
            raise AttributeError(f"'{attr}' is not a valid attribute name.")

    def __add__(self, value: Union[int, float]) -> 'vdata.TemporalDataFrame':
        """
        Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__add__', value)

    def __sub__(self, value: Union[int, float]) -> 'vdata.TemporalDataFrame':
        """
        Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to subtract to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__sub__', value)

    def __mul__(self, value: Union[int, float]) -> 'vdata.TemporalDataFrame':
        """
        Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to multiply all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__mul__', value)

    def __truediv__(self, value: Union[int, float]) -> 'vdata.TemporalDataFrame':
        """
        Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to divide all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__truediv__', value)

    def set(self, values: Any) -> None:
        """
        Set values for this ViewTemporalDataFrame.
        Values can be given as a single value, a collection of values a pandas.DataFrame, a TemporalDataFrame or an
        other ViewTemporalDataFrame.
        To set values from DataFrames, the columns and indexes must match ALL columns and indexes in this
        ViewTemporalDataFrame.
        :param values: new values to set to this view.
        """
        if isinstance(values, pd.DataFrame):
            # only one tp in this view
            # same rows
            # same columns
            assert self.n_time_points == 1
            assert self.columns.equals(values.columns)
            assert self.index.equals(values.index)

            self._parent_data[self.time_points[0]].loc[self.index, self.columns] = values

        elif isinstance(values, (vdata.TemporalDataFrame, ViewTemporalDataFrame)):
            # same tp
            # same rows in all tp
            # same columns in all tp
            assert self.time_points == values.time_points
            assert self.columns.equals(values.columns)
            assert self.index.equals(values.index)

            for tp in self.time_points:
                self._parent_data[tp].loc[self.index_at(tp), self.columns] = values[tp].to_pandas()

        elif isCollection(values):
            values = list(values)
            # only one row or only one column
            if self.n_columns == 1:
                assert len(values) == self.n_index_total, "The length of the index in this view does not match the " \
                                                          "length of the array of values."

                idx_cnt = 0

                for tp in self.time_points:
                    self._parent_data[tp].loc[self.index_at(tp), self.columns] = \
                        values[idx_cnt:idx_cnt+self.n_index_at(tp)]
                    idx_cnt += self.n_index_at(tp)

            elif self.n_index_total == 1:
                assert len(values) == self.n_columns, "The number of columns in this view does not match the length " \
                                                      "of the array of values."
                self._parent_data[self.time_points[0]].loc[self.index] = values

            else:
                raise VValueError(f"Cannot set values in this view with shape {self.shape} from an array of values.")

        else:
            self._parent_data.loc[self.index, self.columns] = values

    def to_pandas(self, with_time_points: Optional[str] = None) -> Any:
        """
        Get the data in a pandas format.
        :param with_time_points: add a column with time points data ?
        :return: the data in a pandas format.
        """
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
    def time_points(self) -> List[TimePoint]:
        """
        Get the list of time points in this view of a TemporalDataFrame.
        :return: the list of time points in this view of a TemporalDataFrame.
        """
        return self._tp_slicer

    @property
    def time_points_column_name(self) -> Optional[str]:
        """
        Get the name of the column with time points data.
        :return: the name of the column with time points data.
        """
        return self._parent.time_points_column_name

    @property
    def index(self):
        """
        Get the full index of this view (concatenated over all time points).
        :return: the full index of this view.
        """
        return self._index

    @index.setter
    def index(self, values: Collection) -> None:
        """
        Set a new index for observations in this view.
        :param values: collection of new index values.
        """
        if not isCollection(values):
            raise VTypeError('New index should be an array of values.')

        len_index = self.n_index_total

        if not len(values) == len_index:
            raise VValueError(f"Cannot reindex from an array of length {len(values)}, should be {len_index}.")

        cnt = 0
        for tp in self.time_points:
            self._parent_data[tp].loc[self.index_at(tp)].index = values[cnt:cnt + self.n_index_at(tp)]
            cnt += self.n_index_at(tp)

    def index_at(self, time_point: TimePoint) -> pd.Index:
        """
        Get the index of this view of a TemporalDataFrame.
        :param time_point: a time point in this view of a TemporalDataFrame.
        :return: the index of this view of a TemporalDataFrame.
        """
        if time_point not in self._tp_slicer:
            raise VValueError(f"TimePoint '{time_point}' cannot be found in this view.")

        return self._parent_data[time_point].index.intersection(self.index, sort=False)

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this view of a TemporalDataFrame.
        :return: the column names of this view of a TemporalDataFrame.
        """
        return self._columns

    @property
    def name(self) -> str:
        """
        Get the name of this view of a TemporalDataFrame.
        :return: the name of this view of a TemporalDataFrame.
        """
        return self._parent.name

    @property
    def dtypes(self) -> None:
        """
        Return the dtypes in the DataFrame.
        :return: the dtypes in the DataFrame.
        """
        return self._parent_data[self.columns].dtypes

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

    def memory_usage(self, index: bool = True, deep: bool = False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of the index and elements of object dtype.
        :return: the memory usage of each column in bytes.
        """
        selected_columns = list(self.columns)
        if index:
            selected_columns.insert(0, 'Index')

        return self._parent.memory_usage(index=index, deep=deep)[selected_columns]

    @property
    def at(self) -> '_ViewVAtIndexer':
        """
        Access a single value for a row/column label pair.
        :return: a single value for a row/column label pair.
        """
        return _ViewVAtIndexer(self._parent, self._parent_data)

    @property
    def iat(self) -> '_ViewViAtIndexer':
        """
        Access a single value for a row/column pair by integer position.
        :return: a single value for a row/column pair by integer position.
        """
        return _ViewViAtIndexer(self._parent, self._parent_data)

    @property
    def loc(self) -> _VLocIndexer:
        """
        Access a group of rows and columns by label(s) or a boolean array.
        :return: a group of rows and columns
        """
        return _VLocIndexer(self._parent, self._parent_data)

    @property
    def iloc(self) -> _ViLocIndexer:
        """
        Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).
        :return: a group of rows and columns
        """
        return _ViLocIndexer(self._parent, self._parent_data)

    def insert(self, *args, **kwargs) -> NoReturn:
        """
        See TemporalDataFrame.insert().
        """
        raise VValueError("Cannot insert a column from a view.")

    def copy(self) -> 'vdata.TemporalDataFrame':
        """
        Create a new copy of this view of a TemporalDataFrame.
        :return: a copy of this view of a TemporalDataFrame.
        """
        return copy.copy_TemporalDataFrame(self)

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

        _name = f"Mean of {self.name}'s view" if self.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def min(self, axis: Literal[0, 1] = 0) -> 'vdata.TemporalDataFrame':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self.name}'s view" if self.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def max(self, axis: Literal[0, 1] = 0) -> 'vdata.TemporalDataFrame':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self.name}'s view" if self.name != 'No_Name' else None
        return vdata.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)


class _ViewVAtIndexer(_VAtIndexer):
    """
    Wrapper around pandas _AtIndexer object for use in views of TemporalDataFrames.
    The .at can access elements by indexing with :
        - a single element (VTDF.loc[<element0>])    --> on indexes

    Allowed indexing elements are :
        - a single label
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        super().__init__(parent, data)

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row index and column name.
        :param value: a value to set.
        """
        # check that data can be found from the given key, to avoid setting data not accessible from the view
        _ = self[key]

        super().__setitem__(key, value)


class _ViewViAtIndexer(_ViAtIndexer):
    """
    Wrapper around pandas _iAtIndexer object for use in views of TemporalDataFrames.
    The .iat can access elements by indexing with :
        - a 2-tuple of elements (TDF.loc[<element0>, <element1>])             --> on indexes and columns

    Allowed indexing elements are :
        - a single integer
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict[TimePoint, pd.DataFrame]):
        """
        :param parent: a parent TemporalDataFrame.
        :param data: the parent TemporalDataFrame's data to work on.
        """
        super().__init__(parent, data)

    def __setitem__(self, key: Tuple[int, int], value: Any) -> None:
        """
        Set values using the _AtIndexer.
        :param key: a tuple of row # and column #.
        :param value: a value to set.
        """
        _ = self[key]

        super().__setitem__(key, value)
