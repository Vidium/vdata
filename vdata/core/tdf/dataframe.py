# coding: utf-8
# Created on 28/03/2022 11:22
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import pandas as pd
import numpy_indexed as npi
from numbers import Number

from typing import Collection, Any, Iterable

from vdata.time_point import TimePoint
from vdata.IO import VLockError
from vdata.core.tdf.name_utils import SLICER
from vdata.core.tdf.utils import parse_slicer, parse_values
from vdata.core.tdf.base import BaseTemporalDataFrameImplementation, BaseTemporalDataFrame
from vdata.core.tdf.view import TemporalDataFrameView
from vdata.core.tdf._parse import parse_data
from vdata.utils import isCollection


# ====================================================
# code
class TemporalDataFrame(BaseTemporalDataFrameImplementation):
    """
    An equivalent to pandas DataFrames that includes the notion of time on the rows.
    This class implements a modified sub-setting mechanism to subset on time points, rows and columns
    """
    _attributes = ('_repeating_index', '_lock', '_timepoints_column_name', '_name', '_timepoint_masks')

    # region magic methods
    def __init__(self,
                 data: dict | pd.DataFrame | None = None,
                 index: Collection | None = None,
                 repeating_index: bool = False,
                 columns_numerical: Collection | None = None,
                 columns_string: Collection | None = None,
                 time_list: Collection[Number | str | TimePoint] | None = None,
                 time_col_name: str | None = None,
                 lock: tuple[bool, bool] | None = None,
                 name: Any = 'No_Name'):
        """
        Args:
            data: Optional object containing the data to store in this TemporalDataFrame. It can be :
                - a dictionary of ['column_name': [values]], where [values] has always the same length
                - a pandas DataFrame
                - a single value to fill the data with
            index: Optional collection of indices. Must match the total number of rows in this TemporalDataFrame,
                over all time-points.
            repeating_index: Is the index repeated at all time-points ?
                If False, the index must contain unique values.
                If True, the index must be exactly equal at all time-points.
            columns_numerical: Optional collection of column names for numerical columns. Must match the number of
                numerical columns in this TemporalDataFrame.
            columns_string: Optional collection of column names for string columns. Must match the number of string
                columns in this TemporalDataFrame.
            time_list: Optional list of time values of the same length as the index, indicating for each row at which
                time point it exists.
            time_col_name: Optional column name in data (if data is a dictionary or a pandas DataFrame) to use as
                time list. This parameter will be ignored if the 'time_list' parameter was set.
            lock: Optional 2-tuple of booleans indicating which axes (index, columns) are locked.
                If 'index' is locked, .index.setter() and .reindex() cannot be used.
                If 'columns' is locked, .__delattr__(), .columns.setter() and .insert() cannot be used.
            name: a name for this TemporalDataFrame.
        """
        _numerical_array, _string_array, _timepoints_array, _index, _columns_numerical, _columns_string, \
            _lock, _timepoints_column_name, _name, repeating_index = \
            parse_data(data, index, repeating_index, columns_numerical, columns_string, time_list, time_col_name, lock,
                       name)

        self._numerical_array = _numerical_array
        self._string_array = _string_array
        self._timepoints_array = _timepoints_array
        self._index = _index
        self._repeating_index = repeating_index
        self._columns_numerical = _columns_numerical
        self._columns_string = _columns_string
        self._lock = _lock
        self._timepoints_column_name = _timepoints_column_name
        self._name = _name

        self._timepoint_masks = dict()

    def __getattr__(self,
                    column_name: str) -> TemporalDataFrameView:
        """
        Get a single column from this TemporalDataFrame.
        """
        if column_name in object.__getattribute__(self, 'columns_num'):
            return TemporalDataFrameView(self, np.arange(len(self.index)), np.array([column_name]), np.array([]))

        elif column_name in object.__getattribute__(self, 'columns_str'):
            return TemporalDataFrameView(self, np.arange(len(self.index)), np.array([]), np.array([column_name]))

        raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")

    def __setattr__(self,
                    name: str,
                    values: np.ndarray) -> None:
        """
        Set values of a single column. If the column does not already exist in this TemporalDataFrame, it is created
            at the end.
        """
        if name in BaseTemporalDataFrameImplementation._attributes or name in TemporalDataFrame._attributes:
            object.__setattr__(self, name, values)
            return

        values = np.array(values)

        if (l := len(values)) != (n := self.n_index):
            raise ValueError(f"Wrong number of values ({l}) for column '{name}', expected {n}.")

        if name in self.columns_num:
            # set values for numerical column
            self.values_num[:, np.where(self.columns_num == name)[0][0]] = \
                values.astype(self.values_num.dtype)

        elif name in self.columns_str:
            # set values for string column
            self._string_array[:, np.where(self.columns_str == name)[0][0]] = values.astype(str)

        else:
            if np.issubdtype(values.dtype, np.number):
                # create numerical column
                self.values_num = np.append(self.values_num,
                                            values.astype(self._numerical_array.dtype)[:, None],
                                            axis=1)

                self._columns_numerical.resize((self.n_columns_num + 1,), refcheck=False)
                self._columns_numerical[-1] = name

            else:
                # create string column
                object.__setattr__(self, '_string_array',
                                   np.append(self._string_array, values.astype(str)[:, None], axis=1))

                self._columns_string.resize((self.n_columns_str + 1,), refcheck=False)
                self._columns_string[-1] = name

    def __delattr__(self,
                    column_name: str) -> None:
        """Drop a column."""
        def drop_column_np(array_: np.ndarray,
                           columns_: np.ndarray,
                           index_: int) -> tuple[np.ndarray, np.ndarray]:
            # delete column from the data array
            array_ = np.delete(array_, index_, 1)

            # delete column from the column names
            columns_ = np.delete(columns_, index_)

            return array_, columns_

        if self.has_locked_columns:
            raise VLockError("Cannot delete column from tdf with locked columns.")

        if column_name in self.columns_num:
            item_index = np.where(self.columns_num == column_name)[0][0]

            self.values_num, self._columns_numerical = \
                drop_column_np(self.values_num, self._columns_numerical, item_index)

        elif column_name in self.columns_str:
            item_index = np.where(self.columns_str == column_name)[0][0]

            self._string_array, self._columns_string = \
                drop_column_np(self._string_array, self._columns_string, item_index)

        else:
            raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")

    def __getitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) -> TemporalDataFrameView:
        index_slicer, column_num_slicer, column_str_slicer, _ = parse_slicer(self, slicer)

        return TemporalDataFrameView(self, index_slicer, column_num_slicer, column_str_slicer)

    def __setitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER],
                    values: Number | np.number | str | Collection | BaseTemporalDataFrame) -> None:
        """
        Set values in a subset.
        """
        # TODO : setattr if setting a single column

        index_positions, column_num_slicer, column_str_slicer, (_, index_array, columns_array) = \
            parse_slicer(self, slicer)

        if columns_array is None:
            columns_array = self.columns

        # parse values
        lcn, lcs = len(column_num_slicer), len(column_str_slicer)

        values = parse_values(values, len(index_positions), lcn + lcs)

        if not lcn + lcs:
            return

        # reorder values to match original index
        if index_array is not None:
            if index_array is None:
                index_array = self.index_at(self.timepoints[0]) if self.has_repeating_index else self.index

            index_positions.sort()

            original_positions = self._get_index_positions(index_array)
            values = values[np.argsort(npi.indices(index_positions,
                                                   original_positions[np.isin(original_positions, index_positions)]))]

        if lcn:
            self.values_num[index_positions[:, None], npi.indices(self._columns_numerical, column_num_slicer)] = \
                values[:, npi.indices(columns_array, column_num_slicer)].astype(float)

        if lcs:
            values_str = values[:, npi.indices(columns_array, column_str_slicer)].astype(str)
            if values_str.dtype > self._string_array.dtype:
                object.__setattr__(self, '_string_array', self._string_array.astype(values_str.dtype))

            self.values_str[index_positions[:, None], npi.indices(self._columns_string, column_str_slicer)] = \
                values_str

    def __invert__(self) -> TemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return TemporalDataFrameView(self, np.arange(0, self.n_index), self.columns_num, self.columns_str,
                                     inverted=True)

    # endregion

    # region attributes
    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @name.setter
    def name(self,
             name: str) -> None:
        """Set the name."""
        object.__setattr__(self, '_name', str(name))

    @property
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """
        unique_timepoints_idx = np.unique(self._timepoints_array, return_index=True)[1]
        return self._timepoints_array[sorted(unique_timepoints_idx)]

    @property
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """
        return self._timepoints_array.copy()

    @property
    def timepoints_column_name(self) -> str | None:
        """
        Get the name of the column containing the time-points values.
        """
        return self._timepoints_column_name

    @property
    def index(self) -> np.ndarray:
        """
        Get the index across all time-points.
        """
        return self._index.copy()

    @property
    def columns_num(self) -> np.ndarray:
        """
        Get the list of column names for numerical data.
        """
        return self._columns_numerical

    @columns_num.setter
    def columns_num(self,
                    values: np.ndarray) -> None:
        """
        Set the list of column names for numerical data.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        if not (vs := values.shape) == (s := self._columns_numerical.shape):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        self._columns_numerical = values

    @property
    def columns_str(self) -> np.ndarray:
        """
        Get the list of column names for string data.
        """
        return self._columns_string

    @columns_str.setter
    def columns_str(self,
                    values: np.ndarray) -> None:
        """
        Set the list of column names for string data.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        if not (vs := values.shape) == (s := self._columns_string.shape):
            raise ValueError(f"Shape mismatch, new 'columns_str' values have shape {vs}, expected {s}.")

        self._columns_string = values

    @property
    def values_num(self) -> np.ndarray:
        """
        Get the numerical data.
        """
        return self._numerical_array

    @values_num.setter
    def values_num(self,
                   values: np.ndarray) -> None:
        """
        Set the numerical data.
        """
        self._numerical_array = values

    @property
    def values_str(self) -> np.ndarray:
        """
        Get the string data.
        """
        return self._string_array

    @values_str.setter
    def values_str(self,
                   values: np.ndarray) -> None:
        """
        Set the string data.
        """
        if values.dtype.type != np.str_:
            values = values.astype(str)

        self._string_array = values

    # endregion

    # region predicates
    @property
    def has_locked_indices(self) -> bool:
        """
        Is the "index" axis locked for modification ?
        """
        return bool(self._lock[0])

    @property
    def has_locked_columns(self) -> bool:
        """
        Is the "columns" axis locked for modification ?
        """
        return bool(self._lock[1])

    @property
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """
        return self._repeating_index

    # endregion

    # region methods
    def _fast_compare(self,
                      comparison_tp: TimePoint | bytes) -> np.ndarray:
        if not (ltpm := len(self._timepoint_masks)):
            return np.equal(self._timepoints_array, comparison_tp)

        tp_mask = np.zeros(len(self._timepoints_array), dtype=bool)

        if ltpm == 1:
            not_already_computed = ~next(iter(self._timepoint_masks.values()))

        else:
            not_already_computed = np.logical_and.reduce([~mask for mask in self._timepoint_masks.values()])

        tp_mask[not_already_computed] = np.equal(self._timepoints_array[not_already_computed], comparison_tp)
        return tp_mask

    def get_timepoint_mask(self,
                           timepoint: str | TimePoint) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """
        # cache masks for performance, cache is reinitialized when _timepoints_array changes
        if timepoint not in self._timepoint_masks.keys():
            comparison_tp = TimePoint(timepoint)
            self._timepoint_masks[timepoint] = self._fast_compare(comparison_tp)

        return self._timepoint_masks[timepoint]

    def lock_indices(self) -> None:
        object.__setattr__(self, '_lock', (True, self.has_locked_columns))

    def unlock_indices(self) -> None:
        object.__setattr__(self, '_lock', (False, self.has_locked_columns))

    def lock_columns(self) -> None:
        object.__setattr__(self, '_lock', (self.has_locked_indices, True))

    def unlock_columns(self) -> None:
        object.__setattr__(self, '_lock', (self.has_locked_indices, False))

    def set_index(self,
                  values: np.ndarray,
                  repeating_index: bool = False) -> None:
        """Set new index values."""
        if self.has_locked_indices:
            raise VLockError("Cannot set index in tdf with locked index.")

        self._check_valid_index(values, repeating_index)

        object.__setattr__(self, '_index', values)
        object.__setattr__(self, '_repeating_index', repeating_index)

    def reindex(self,
                order: np.ndarray,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        super().reindex(order, repeating_index)

        # set index
        object.__setattr__(self, '_index', order)
        object.__setattr__(self, '_repeating_index', repeating_index)

    def insert(self,
               loc: int,
               name: str,
               values: np.ndarray | Iterable | int | float | str) -> None:
        """
        Insert a column in either the numerical data or the string data, depending on the type of the <values> array.
            The column is inserted at position <loc> with name <name>.
        """
        def insert_column_np(array_: np.ndarray,
                             columns_: np.ndarray,
                             index_: int) -> tuple[np.ndarray, np.ndarray]:
            if index_ < 0:
                index_ = len(columns_) + 1 + index_

            # insert column in the data array the position index_.
            array_ = np.insert(array_, index_, values, axis=1)

            # insert column in the column names
            columns_ = np.insert(columns_, index_, name)

            return array_, columns_

        if self.has_locked_columns:
            raise VLockError("Cannot insert columns in tdf with locked columns.")

        if not isCollection(values):
            values = np.repeat(values, self.n_index)

        values = np.array(values)

        if (l := len(values)) != (n := self.n_index):
            raise ValueError(f"Wrong number of values ({l}), expected {n}.")

        if name in self.columns:
            raise ValueError(f"A column named '{name}' already exists.")

        if np.issubdtype(values.dtype, np.number):
            # create numerical column
            array, columns = insert_column_np(self.values_num, self._columns_numerical, loc)
            self.values_num = array
            object.__setattr__(self, '_columns_numerical', columns)

        else:
            # create string column
            array, columns = insert_column_np(self._string_array, self._columns_string, loc)
            object.__setattr__(self, '_string_array', array)
            object.__setattr__(self, '_columns_string', columns)

    # endregion
