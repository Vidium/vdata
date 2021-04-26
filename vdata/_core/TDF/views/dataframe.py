# coding: utf-8
# Created on 15/01/2021 12:57
# Author : matteo

# ====================================================
# imports
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Collection, Optional, Union, Tuple, Any, Dict, List, NoReturn
from typing_extensions import Literal

from ..name_utils import ViewTemporalDataFrame_internal_attributes
from .. import dataframe
from .. import base
from .. import copy
from ..indexers import _VAtIndexer, _ViAtIndexer, _VLocIndexer, _ViLocIndexer
from ...name_utils import PreSlicer
from ...utils import reformat_index, repr_index
from vdata.name_utils import DType
from vdata.utils import repr_array, isCollection
from vdata.time_point import TimePoint
from ...._IO import generalLogger, VValueError, VAttributeError, VTypeError, VLockError


# ==========================================
# code
class ViewTemporalDataFrame(base.BaseTemporalDataFrame):
    """
    A view of a TemporalDataFrame, created on sub-setting operations.
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', parent_data: Dict['TimePoint', np.ndarray],
                 tp_slicer: Collection['TimePoint'], index_slicer: Collection, column_slicer: Collection,
                 lock: bool):
        """
        :param parent: a parent TemporalDataFrame to view.
        :param parent_data: the parent TemporalDataFrame's data.
        :param tp_slicer: a collection of time points to view.
        :param index_slicer: a pandas Index of rows to view.
            obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
        :param column_slicer: a pandas Index of columns to view.
        :param lock: lock this view on index modification ?
        """
        generalLogger.debug(f"\u23BE ViewTemporalDataFrame '{parent.name}':{id(self)} creation : begin "
                            f"---------------------------------------- ")

        # set attributes on init using object's __setattr__ method to avoid self's __setattr__ which would provoke bugs
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_parent_data', parent_data)
        object.__setattr__(self, '_index', pd.Index(index_slicer))
        object.__setattr__(self, '_columns', pd.Index([c for c in parent.columns if c in column_slicer]))

        # remove time points where the index does not match
        object.__setattr__(self, '_tp_slicer', sorted(np.array(tp_slicer)[
            [any(self.index.isin(parent.index_at(time_point))) for time_point in tp_slicer]]))

        generalLogger.debug(f"  1. Refactored time point slicer to : {repr_array(self._tp_slicer)}")

        # remove index elements where time points do not match
        if len(self._tp_slicer):
            valid_indexes = np.concatenate([parent.index_at(time_point) for time_point in self._tp_slicer])
            index_at_tp_slicer = pd.Index([i for i in valid_indexes if i in self.index])

        else:
            index_at_tp_slicer = pd.Index([], dtype=object)

        object.__setattr__(self, '_index', index_at_tp_slicer)

        generalLogger.debug(f"  2. Refactored index slicer to : {repr_array(self.index)}")

        object.__setattr__(self, '_lock', lock)

        generalLogger.debug(f"\u23BF ViewTemporalDataFrame '{parent.name}':{id(self)} creation : end "
                            f"------------------------------------------ ")

    def __repr__(self):
        """
        Description for this view of a TemporalDataFrame object to print.
        :return: a description of this view of a TemporalDataFrame object.
        """
        if not self.empty:
            repr_str = f"View of TemporalDataFrame '{self.name}'\n"

        else:
            repr_str = f"Empty View of TemporalDataFrame '{self.name}'\n"

        repr_str += self._head()

        return repr_str

    def __getitem__(self, index: Union['PreSlicer',
                                       Tuple['PreSlicer'],
                                       Tuple['PreSlicer', 'PreSlicer'],
                                       Tuple['PreSlicer', 'PreSlicer', 'PreSlicer']]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a sub-view from this view using an index with the usual sub-setting mechanics.
        :param index: A sub-setting index.
            See TemporalDataFrame's '__getitem__' method for more details.
        """
        generalLogger.debug(f"ViewTemporalDataFrame '{self.name}':{id(self)} sub-setting "
                            f"- - - - - - - - - - - - - -")
        generalLogger.debug(f'  Got index : {repr_index(index)}')

        if isinstance(index, tuple) and len(index) == 3 and not isCollection(index[2]) \
                and not isinstance(index[2], slice) and index[2] is not ... \
                and (index[0] is ... or index[0] == slice(None))\
                and (index[1] is ... or index[1] == slice(None)):
            return self.__getattr__(index[2])

        else:
            _index = reformat_index(index, self.time_points, self.index, self.columns)

            generalLogger.debug(f'  Refactored index to : {repr_index(_index)}')

            if not len(_index[0]):
                raise VValueError("Time point not found.")

            return ViewTemporalDataFrame(self._parent, self._parent_data, _index[0], _index[1], _index[2],
                                         self._lock)

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
        if attr in self.columns:
            return self.loc[self.index, attr]

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

    def __add__(self, value: Union[int, float]) -> 'dataframe.TemporalDataFrame':
        """
        Add an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to add to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__add__', value)

    def __sub__(self, value: Union[int, float]) -> 'dataframe.TemporalDataFrame':
        """
        Subtract an int or a float to all values in this TemporalDataFrame and return a new TemporalDataFrame.
        :param value: an int or a float to subtract to values.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__sub__', value)

    def __mul__(self, value: Union[int, float]) -> 'dataframe.TemporalDataFrame':
        """
        Multiply all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to multiply all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__mul__', value)

    def __truediv__(self, value: Union[int, float]) -> 'dataframe.TemporalDataFrame':
        """
        Divide all values in this TemporalDataFrame by an int or a float and return a new TemporalDataFrame.
        :param value: an int or a float to divide all values by.
        :return: a TemporalDataFrame with new values.
        """
        return self._asmd_func('__truediv__', value)

    def __eq__(self, other):
        if isinstance(other, (dataframe.TemporalDataFrame, ViewTemporalDataFrame)):
            return self.time_points == other.time_points and self.index == other.index and self.columns == \
                   other.columns and all([self._df[tp] == other._df[tp] for tp in self.time_points])

        elif self.n_columns == 1:
            return self.eq(other).values.flatten()

        else:
            return self.eq(other)

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

            col_indices = np.where(self.bool_columns())[0]

            for col_index_in_values, col_index in enumerate(col_indices):
                self._parent_data[self.time_points[0]][np.where(self.bool_index_at(self.time_points[0]))[0],
                                                       col_index] = values.iloc[:, col_index_in_values]

        elif isinstance(values, (dataframe.TemporalDataFrame, ViewTemporalDataFrame)):
            # same tp
            # same rows in all tp
            # same columns in all tp
            assert self.time_points == values.time_points
            assert self.columns.equals(values.columns)
            assert self.index.equals(values.index)

            for tp in self.time_points:
                pandas_values = values[tp].to_pandas()

                col_indices = np.where(self.bool_columns())[0]

                for col_index_in_values, col_index in enumerate(col_indices):
                    self._parent_data[tp][np.where(self.bool_index_at(tp))[0], col_index] = \
                        pandas_values.iloc[:, col_index_in_values]

        elif isCollection(values):
            values = list(values)
            # only one row or only one column
            if self.n_columns == 1:
                assert len(values) == self.n_index_total, f"The length of the index in this view " \
                                                          f"({self.n_index_total}) does not match the length of the " \
                                                          f"array of values ({len(values)})."

                idx_cnt = 0

                for time_point in self.time_points:
                    col_index = np.where(self.bool_columns())[0][0]

                    self._parent_data[time_point][np.where(self.bool_index_at(time_point))[0],
                                                  col_index] = values[idx_cnt:idx_cnt+self.n_index_at(time_point)]
                    idx_cnt += self.n_index_at(time_point)

            elif self.n_index_total == 1:
                assert len(values) == self.n_columns, f"The number of columns in this view ({self.n_columnsl}) does " \
                                                      f"not match the length of the array of values ({len(values)})."

                col_indices = np.where(self.bool_columns())[0]
                for col_index in col_indices:
                    self._parent_data[self.time_points[0]][np.where(self.bool_index_at(self.time_points[0]))[0],
                                                           col_index] = values

            else:
                raise VValueError(f"Cannot set values in this view with shape {self.shape} from an array of values.")

        else:
            # single value for all data
            for time_point in self.time_points:
                col_indices = np.where(self.bool_columns())[0]

                for col_index in col_indices:
                    self._parent_data[time_point][np.where(self.bool_index_at(time_point))[0], col_index] \
                        = values

    @property
    def is_locked(self) -> Tuple[bool, bool]:
        """
        Get this view of a TemporalDataFrame's lock.
        This controls what can be modified with 2 boolean values :
            1. True --> cannot use self.index.setter() and self.reindex()
            2. True (always) --> cannot use self.__delattr__(), self.columns.setter() and self.insert()
        """
        return self._lock, True

    def to_pandas(self, with_time_points: bool = False) -> Any:
        """
        Get the data in a pandas format.
        :param with_time_points: add a column with time points data ?
        :return: the data in a pandas format.
        """
        if not self.empty:
            data = pd.concat([
                pd.DataFrame(self._parent_data[time_point][np.where(self.bool_index_at(time_point))[0]][:,
                             np.where(self.bool_columns())[0]],
                             index=self.index_at(time_point),
                             columns=self.columns)
                for time_point in self.time_points if self._parent_data[time_point].ndim == 2])

        else:
            data = pd.DataFrame(index=self.index, columns=self.columns)

        if with_time_points:
            if self.time_points_column_name is not None:
                data.insert(0, self.time_points_column_name, self.time_points_column)

            else:
                data.insert(0, 'Time_Point', self.time_points_column)

        return data

    @property
    def time_points(self) -> List['TimePoint']:
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
        if self._lock:
            raise VLockError("Cannot use 'index.setter' functionality on a locked view of a TemporalDataFrame.")

        else:
            if not isCollection(values):
                raise VTypeError('New index should be an array of values.')

            len_index = self.n_index_total

            if not len(values) == len_index:
                raise VValueError(f"Cannot reindex from an array of length {len(values)}, should be {len_index}.")

            cnt = 0
            for tp in self.time_points:
                self._parent_data[tp].loc[self.index_at(tp)].index = values[cnt:cnt + self.n_index_at(tp)]
                cnt += self.n_index_at(tp)

            if self._parent.is_backed and self._parent.file.file.mode == 'r+':
                self._parent.file['index'][()] = self._parent.index
                self._parent.file.file.flush()

    def index_at(self, time_point: Union['TimePoint', str]) -> pd.Index:
        """
        Get the index of this view of a TemporalDataFrame.
        :param time_point: a time point in this view of a TemporalDataFrame.
        :return: the index of this view of a TemporalDataFrame.
        """
        if not isinstance(time_point, TimePoint):
            time_point = TimePoint(time_point)

        if time_point not in self._tp_slicer:
            raise VValueError(f"TimePoint '{time_point}' cannot be found in this view.")

        dup_index = self._parent.index_at(time_point).intersection(self.index, sort=False)
        unique_idx = np.unique(dup_index, return_index=True)[1]

        return pd.Index([dup_index[i] for i in sorted(unique_idx)])

    def bool_index_at(self, time_point: Union['TimePoint', str]) -> np.ndarray:
        """
        Get a boolean array of the same length as the parental index, where True means that the index at that
        position is in this view.
        :return: boolean array on parental index in this view.
        """
        if not isinstance(time_point, TimePoint):
            time_point = TimePoint(time_point)

        if time_point in self.time_points:
            return self._parent.index_at(time_point).isin(self.index_at(time_point))

        else:
            return np.array([False for _ in range(self._parent.n_index_at(time_point))])

    @property
    def columns(self) -> pd.Index:
        """
        Get the columns of this view of a TemporalDataFrame.
        :return: the column names of this view of a TemporalDataFrame.
        """
        return self._columns

    def bool_columns(self) -> np.ndarray:
        """
        Get a boolean array of the same length as the parental columns index, where True means that the column at that
        position is in this view.
        :return: boolean array on parental columns index in this view.
        """
        return self._parent.columns.isin(self.columns)

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
        return self._parent_data[self.columns].dtype

    def astype(self, dtype: Union['DType', Dict[str, 'DType']]) -> NoReturn:
        """
        Reference to TemporalDataFrame's astype method. This cannot be done in a view.
        """
        raise VAttributeError('Cannot set data type from a view of a TemporalDataFrame.')

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
        return _VLocIndexer(self, self._parent_data)

    @property
    def iloc(self) -> _ViLocIndexer:
        """
        Purely integer-location based indexing for selection by position (from 0 to length-1 of the axis).
        :return: a group of rows and columns
        """
        return _ViLocIndexer(self, self._parent_data)

    def insert(self, *args, **kwargs) -> NoReturn:
        """
        See TemporalDataFrame.insert().
        """
        raise VValueError("Cannot insert a column from a view.")

    def copy(self) -> 'dataframe.TemporalDataFrame':
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
            _data = {func: np.concatenate(
                [getattr(np.array(self._parent_data[tp])[self.bool_index_at(tp)][:, self.bool_columns()],
                         func)(axis=0) for tp in self.time_points]
            )}
            _time_list = np.repeat(self.time_points, self.n_columns)
            _index = pd.Index(np.concatenate([self.columns for _ in range(self.n_time_points)]))

        elif axis == 1:
            _data = {func: np.concatenate(
                [getattr(np.array(self._parent_data[tp])[self.bool_index_at(tp)][:, self.bool_columns()],
                         func)(axis=1) for tp in self.time_points]
            )}
            _time_list = self.time_points_column
            _index = self.index

        else:
            raise VValueError(f"Invalid axis '{axis}', should be 0 (on columns) or 1 (on rows).")

        return _data, _time_list, _index

    def mean(self, axis: Literal[0, 1] = 0) -> 'dataframe.TemporalDataFrame':
        """
        Return the mean of the values over the requested axis.

        :param axis: compute mean over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with mean values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('mean', axis)

        _name = f"Mean of {self.name}'s view" if self.name != 'No_Name' else None
        return dataframe.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def min(self, axis: Literal[0, 1] = 0) -> 'dataframe.TemporalDataFrame':
        """
        Return the minimum of the values over the requested axis.

        :param axis: compute minimum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with minimum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('min', axis)

        _name = f"Minimum of {self.name}'s view" if self.name != 'No_Name' else None
        return dataframe.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def max(self, axis: Literal[0, 1] = 0) -> 'dataframe.TemporalDataFrame':
        """
        Return the maximum of the values over the requested axis.

        :param axis: compute maximum over columns (0: default) or over rows (1).
        :return: a TemporalDataFrame with maximum values.
        """
        _data, _time_list, _index = self.__mean_min_max_func('max', axis)

        _name = f"Maximum of {self.name}'s view" if self.name != 'No_Name' else None
        return dataframe.TemporalDataFrame(_data, time_list=_time_list, index=_index, name=_name)

    def write(self, file: Union[str, Path]) -> None:
        """
        Save this TemporalDataFrame in HDF5 file format.

        :param file: path to save the TemporalDataFrame.
        """
        from ...._read_write import write_TemporalDataFrame

        with h5py.File(file, 'w') as save_file:
            write_TemporalDataFrame(self.copy(), save_file, self.name)


class _ViewVAtIndexer(_VAtIndexer):
    """
    Wrapper around pandas _AtIndexer object for use in views of TemporalDataFrames.
    The .at can access elements by indexing with :
        - a single element (VTDF.loc[<element0>])    --> on indexes

    Allowed indexing elements are :
        - a single label
    """

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict['TimePoint', np.ndarray]):
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

    def __init__(self, parent: 'dataframe.TemporalDataFrame', data: Dict['TimePoint', np.ndarray]):
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
