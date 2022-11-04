# coding: utf-8
# Created on 17/10/2022 19:10
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from itertools import chain

import numpy as np
import numpy_indexed as npi
from numbers import Number

from typing import Collection, Iterable

from vdata.IO import VLockError
from vdata.h5pickle.name_utils import H5Mode
from vdata.time_point import TimePoint
from vdata.core.dataset_proxy import DatasetProxy
from vdata.core.tdf.backed_tdf._parse import parse_data_h5
from vdata.core.tdf.backed_tdf.base import BackedMixin
from vdata.core.tdf.backed_tdf.view import BackedTemporalDataFrameView
from vdata.core.tdf.base import BaseTemporalDataFrameImplementation, BaseTemporalDataFrame
from vdata.core.tdf.name_utils import H5Data, SLICER
from vdata.core.tdf.utils import parse_slicer, parse_values
from vdata.utils import isCollection

# ====================================================
# code
_CHECK_READ = ('__getattr__', '__getitem__', '__invert__', 'timepoints', 'timepoints_column',
               'timepoints_column_name', 'file', 'has_locked_indices', 'has_locked_columns', 'has_repeating_index')
_CHECK_WRITE = ('__setattr__', '__delattr__', '__setitem__', 'lock_indices', 'unlock_indices', 'lock_columns',
                'unlock_columns', 'set_index', 'reindex', 'insert')


class BackedTemporalDataFrame(BackedMixin, BaseTemporalDataFrameImplementation,
                              read=_CHECK_READ, write=_CHECK_WRITE):
    """
    A version of the TemporalDataFrame backed on a h5 file for RAM performance.
    """
    _attributes = ('_file', '_attributes', 'dataset_num', 'dataset_str')

    # region magic methods
    def __init__(self,
                 data: H5Data,
                 lock: tuple[bool, bool] | None = None,
                 name: str = 'No_Name'):
        """
        Args:
            data: a H5 File or Group containing numerical and string data.
            lock: Optional 2-tuple of booleans indicating which axes (index, columns) are locked.
                If 'index' is locked, .index.setter() and .reindex() cannot be used.
                If 'columns' is locked, .__delattr__(), .columns.setter() and .insert() cannot be used.
            name: a name for this TemporalDataFrame.
        """
        super().__init__()

        # parse h5 file
        _numerical_array, _string_array, _index, _columns_numerical, _columns_string, _timepoints_array, _attributes \
            = parse_data_h5(data, lock, name)

        self._file = data
        self._numerical_array = _numerical_array
        self._string_array = _string_array
        self._index = _index
        self._columns_numerical = _columns_numerical
        self._columns_string = _columns_string
        self._timepoints_array = _timepoints_array
        self._attributes = _attributes

    def __repr__(self) -> str:
        if self.is_closed:
            return self.full_name

        return super().__repr__()

    def __dir__(self) -> Iterable[str]:
        return chain(dir(super()), dir(BackedMixin))

    def __getattr__(self,
                    column_name: str) -> BackedTemporalDataFrameView:
        """
        Get a single column from this TemporalDataFrame.
        """
        n_index = len(object.__getattribute__(self, '_index'))

        if column_name in object.__getattribute__(self, 'columns_num'):
            return BackedTemporalDataFrameView(self, np.arange(n_index), np.array([column_name]), np.array([]))

        elif column_name in object.__getattribute__(self, 'columns_str'):
            return BackedTemporalDataFrameView(self, np.arange(n_index), np.array([]), np.array([column_name]))

        raise AttributeError(f"'{column_name}' not found in this backed TemporalDataFrame.")

    def __setattr__(self,
                    name: str,
                    values: np.ndarray) -> None:
        """
        Set values of a single column. If the column does not already exist in this TemporalDataFrame, it is created
            at the end.
        """
        if name in BaseTemporalDataFrameImplementation._attributes or name in BackedTemporalDataFrame._attributes:
            object.__setattr__(self, name, values)
            return

        values = np.array(values)

        if (l := len(values)) != (n := self.n_index):
            raise ValueError(f"Wrong number of values ({l}) for column '{name}', expected {n}.")

        if name in self.columns_num:
            # set values for numerical column
            self.dataset_num[:, np.where(self.columns_num == name)[0][0]] = values

        elif name in self.columns_str:
            # set values for string column
            self.dataset_str[:, np.where(self.columns_str == name)[0][0]] = values

        else:
            if np.issubdtype(values.dtype, np.number):
                # create numerical column
                self.dataset_num.resize((self.n_index, self.n_columns_num + 1))
                self.dataset_num[:, -1] = values

                self.columns_num.resize((self.n_columns_num + 1,))
                self.columns_num[-1] = name

            else:
                # create string column
                self.dataset_str.resize((self.n_index, self.n_columns_str + 1))
                self.dataset_str[:, -1] = values

                self.columns_str.resize((self.n_columns_str + 1,))
                self.columns_str[-1] = name

    def __delattr__(self,
                    column_name: str) -> None:
        """Drop a column."""
        def drop_column_h5(array_: DatasetProxy,
                           columns_: DatasetProxy,
                           index_: int) -> None:
            # transfer data one row to the left, starting from the column after the one to delete
            # matrix | 0 1 2 3 4 | with index of the column to delete = 2
            #   ==>  | 0 1 3 4 . |
            array_[:, index_:len(columns_) - 1] = array_[:, (index_ + 1):len(columns_)]

            # delete column from the column names as above
            columns_[index_:len(columns_) - 1] = columns_[index_ + 1:len(columns_)]

            # resize the arrays to drop the extra column at the end
            columns_.resize((len(columns_) - 1,))
            array_.resize((array_.shape[0], array_.shape[1] - 1))

        if self.has_locked_columns:
            raise VLockError("Cannot delete column from tdf with locked columns.")

        if column_name in self.columns_num:
            item_index = np.where(self.columns_num == column_name)[0][0]

            drop_column_h5(self.dataset_num, self.columns_num, item_index)

        elif column_name in self.columns_str:
            item_index = np.where(self.columns_str == column_name)[0][0]

            drop_column_h5(self.dataset_str, self.columns_str, item_index)

        else:
            raise AttributeError(f"'{column_name}' not found in this backed TemporalDataFrame.")

    def __getitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) \
            -> BackedTemporalDataFrameView:
        index_slicer, column_num_slicer, column_str_slicer, _ = parse_slicer(self, slicer)

        return BackedTemporalDataFrameView(self, index_slicer, column_num_slicer, column_str_slicer)

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
        if index_array is None:
            index_array = self.index_at(self.timepoints[0]) if self.has_repeating_index else self.index

        index_positions.sort()

        original_positions = self._get_index_positions(index_array)
        values = values[np.argsort(npi.indices(index_positions,
                                               original_positions[np.isin(original_positions, index_positions)]))]

        if lcn:
            self.dataset_num[index_positions, npi.indices(self.columns_num[:], column_num_slicer)] = \
                values[:, npi.indices(columns_array, column_num_slicer)].astype(float)

        if lcs:
            self.dataset_str[index_positions, npi.indices(self.columns_str[:], column_str_slicer)] = \
                values[:, npi.indices(columns_array, column_str_slicer)].astype(str)

    def __invert__(self) -> BackedTemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return BackedTemporalDataFrameView(self, np.arange(0, self.n_index), self.columns_num[:], self.columns_str[:],
                                           inverted=True)

    # endregion

    # region attributes
    @property
    def name(self) -> str:
        """Get the name."""
        return self._attributes['name']

    @name.setter
    def name(self,
             name: str) -> None:
        """Set the name."""
        self._attributes['name'] = name

    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        if self.is_closed:
            return "TemporalDataFrame backed on closed file"

        return super().full_name

    @property
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """
        return self._timepoints_array.unique()

    @property
    def timepoints_column(self) -> np.array:
        """
        Get the column of time-point values.
        """
        return self._timepoints_array[:]

    @property
    def timepoints_column_name(self) -> str | None:
        """
        Get the name of the column containing the time-points values.
        """
        return self._attributes['timepoints_column_name']

    @property
    def index(self) -> DatasetProxy:
        """
        Get the index across all time-points.
        """
        return self._index[:]

    @property
    def columns_num(self) -> DatasetProxy:
        """
        Get the list of column names for numerical data.
        """
        return self._columns_numerical

    @property
    def columns_str(self) -> DatasetProxy:
        """
        Get the list of column names for string data.
        """
        return self._columns_string

    @property
    def values_num(self) -> np.ndarray:
        """
        Get the numerical data.
        """
        return self._numerical_array[:]

    @values_num.setter
    def values_num(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the numerical data.
        """
        if isinstance(values, DatasetProxy):
            self._numerical_array = values
            return

        self._numerical_array[:] = values

    @property
    def values_str(self) -> np.ndarray:
        """
        Get the string data.
        """
        return self._string_array[:]

    @values_str.setter
    def values_str(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the string data.
        """
        if isinstance(values, DatasetProxy):
            self._string_array = values
            return

        self._string_array[:] = values

    @property
    def h5_mode(self) -> H5Mode:
        """Get the mode the h5 file was opened with."""
        return self._file.file.mode

    # endregion

    # region predicates
    @property
    def has_locked_indices(self) -> bool:
        """
        Is the "index" axis locked for modification ?
        """
        return bool(self._attributes['locked_indices'])

    @property
    def has_locked_columns(self) -> bool:
        """
        Is the "columns" axis locked for modification ?
        """
        return bool(self._attributes['locked_columns'])

    @property
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """
        return self._attributes['repeating_index']

    @property
    def is_closed(self) -> bool:
        """
        Is the h5 file (this TemporalDataFrame is backed on) closed ?
        """
        return not self._file.file.id.valid

    # endregion

    # region methods
    def get_timepoint_mask(self,
                           timepoint: str | TimePoint) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """
        return self._timepoints_array.is_equal(timepoint)

    def lock_indices(self) -> None:
        self._attributes['locked_indices'] = True

    def unlock_indices(self) -> None:
        self._attributes['locked_indices'] = False

    def lock_columns(self) -> None:
        self._attributes['locked_columns'] = True

    def unlock_columns(self) -> None:
        self._attributes['locked_columns'] = False

    def set_index(self,
                  values: np.ndarray,
                  repeating_index: bool = False) -> None:
        """Set new index values."""
        if self.has_locked_indices:
            raise VLockError("Cannot set index in tdf with locked index.")

        self._check_valid_index(values, repeating_index)

        if np.issubdtype(values.dtype, np.number) and not np.issubdtype(self._index.dtype, np.number):
            self._index.astype(values.dtype, replacement_data=values)

        # no type change
        else:
            self._index[:] = values

        self._attributes['repeating_index'] = repeating_index

    def reindex(self,
                order: np.ndarray,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        super().reindex(order, repeating_index)

        # set index
        self._index[:] = order
        self._attributes['repeating_index'] = repeating_index

    def insert(self,
               loc: int,
               name: str,
               values: np.ndarray | Iterable | int | float) -> None:
        """
        Insert a column in either the numerical data or the string data, depending on the type of the <values> array.
            The column is inserted at position <loc> with name <name>.
        """
        def insert_column_h5(array_: DatasetProxy,
                             columns_: DatasetProxy,
                             index_: int) -> None:
            if index_ < 0:
                index_ = len(columns_) + 1 + index_

            # resize the arrays to insert an extra column at the end
            columns_.resize((len(columns_) + 1,))
            array_.resize((array_.shape[0], array_.shape[1] + 1))

            # transfer data one row to the right, starting from the column after the index
            # matrix | 0 1 2 3 4 | with index = 2
            #   ==>  | 0 1 . 2 3 4 |
            array_[:, index_ + 1:len(columns_)] = array_[:, index_:len(columns_) - 1]

            # insert values at index
            array_[:, index_] = values

            # insert column from the column names as above
            columns_[index_ + 1:len(columns_)] = columns_[index_:len(columns_) - 1]
            columns_[index_] = name

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
            insert_column_h5(self.dataset_num, self.columns_num, loc)

        else:
            # create string column
            insert_column_h5(self.dataset_str, self.columns_str, loc)

    def close(self) -> None:
        """Close the file this TemporalDataFrame is backed on."""
        self._file.close()

    # endregion
