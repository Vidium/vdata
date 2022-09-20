# coding: utf-8
# Created on 28/03/2022 11:22
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import numpy_indexed as npi
from h5py import File, Dataset
from pathlib import Path
from numbers import Number

from typing import Collection, Any, Iterable, Literal

from vdata.time_point import TimePoint
from vdata.name_utils import H5Mode
from vdata.utils import repr_array
from vdata.IO import VLockError
from .name_utils import H5Data, SLICER, DEFAULT_TIME_POINTS_COL_NAME
from .utils import parse_slicer, parse_values
from .base import BaseTemporalDataFrame
from .indexer import VAtIndexer, ViAtIndexer, VLocIndexer, ViLocIndexer
from .view import ViewTemporalDataFrame
from ._parse import parse_data
from ._write import write_TDF, write_array


# ====================================================
# code
def check_can_read(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        file = object.__getattribute__(self, '_file')

        # if TDF is backed but file is closed : can't read
        if file is not None and not file:
            raise ValueError("Can't read TemporalDataFrame backed on closed file.")

        return func(*args, **kwargs)

    return wrapper


def check_can_write(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        file = object.__getattribute__(self, '_file')

        # if TDF is backed but file is closed or file mode is not a or r+ : can't write
        if file is not None:
            if not file:
                raise ValueError("Can't write to TemporalDataFrame backed on closed file.")

            if (m := file.file.mode) not in (H5Mode.READ_WRITE_CREATE, H5Mode.READ_WRITE):
                raise ValueError(f"Can't write to TemporalDataFrame backed on file with mode='{m}'.")

        return func(*args, **kwargs)

    return wrapper


class TemporalDataFrame(BaseTemporalDataFrame):
    """
    An equivalent to pandas DataFrames that includes the notion of time on the rows.
    This class implements a modified sub-setting mechanism to subset on time points, rows and columns
    """

    __slots__ = '_name', '_file', '_numerical_array', '_string_array', '_timepoints_array', '_index', \
                '_repeating_index', '_columns_numerical', '_columns_string', '_lock', '_timepoints_column_name', \
                '_timepoint_masks'

    __attributes = ('name', 'timepoints_column_name', 'index', 'columns', 'columns_num', 'columns_str', 'values_num',
                    'values_str')

    def __init__(self,
                 data: dict | pd.DataFrame | H5Data | None = None,
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
                - a H5 File or Group containing numerical and string data.
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
        # (potentially) backed data :
        #   numerical data (assumed float or int)
        #   string data (objects which are not numbers, assumed strings)
        #   index (any type)
        #   columns (any type)
        _file, _numerical_array, _string_array, _timepoints_array, _index, _columns_numerical, _columns_string, \
            _lock, _timepoints_column_name, _name, repeating_index = \
            parse_data(data, index, repeating_index, columns_numerical, columns_string, time_list, time_col_name, lock,
                       name)

        object.__setattr__(self, '_file', _file)
        object.__setattr__(self, '_numerical_array', _numerical_array)
        object.__setattr__(self, '_string_array', _string_array)
        object.__setattr__(self, '_timepoints_array', _timepoints_array)
        object.__setattr__(self, '_index', _index)
        object.__setattr__(self, '_repeating_index', repeating_index)
        object.__setattr__(self, '_columns_numerical', _columns_numerical)
        object.__setattr__(self, '_columns_string', _columns_string)
        object.__setattr__(self, '_lock', _lock)
        object.__setattr__(self, '_timepoints_column_name', _timepoints_column_name)
        object.__setattr__(self, '_name', _name)
        object.__setattr__(self, '_timepoint_masks', dict())

    @check_can_read
    def __repr__(self) -> str:
        if self.is_backed and not self._file:
            return self.full_name

        return f"{self.full_name}\n{self.head()}"

    @check_can_read
    def __dir__(self) -> Iterable[str]:
        return dir(TemporalDataFrame) + list(map(str, self.columns))

    @check_can_read
    def __getattr__(self,
                    column_name: str) -> ViewTemporalDataFrame:
        """
        Get a single column from this TemporalDataFrame.
        """
        if column_name in self.columns_num:
            return ViewTemporalDataFrame(self, np.arange(len(self.index)), np.array([column_name]), np.array([]))

        elif column_name in self.columns_str:
            return ViewTemporalDataFrame(self, np.arange(len(self.index)), np.array([]), np.array([column_name]))

        raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")

    @check_can_write
    def __setattr__(self,
                    name: str,
                    values: np.ndarray) -> None:
        """
        Set values of a single column. If the column does not already exist in this TemporalDataFrame, it is created
            at the end.
        """
        if name in TemporalDataFrame.__attributes or name in TemporalDataFrame.__slots__:
            object.__setattr__(self, name, values)
            return

        values = np.array(values)

        if (l := len(values)) != (n := self.n_index):
            raise ValueError(f"Wrong number of values ({l}) for column '{name}', expected {n}.")

        if name in self.columns_num:
            # set values for numerical column
            self._numerical_array[:, np.where(self.columns_num == name)[0][0]] = \
                values.astype(self._numerical_array.dtype)

        elif name in self.columns_str:
            # set values for string column
            self._string_array[:, np.where(self.columns_str == name)[0][0]] = values.astype(str)

        else:
            if np.issubdtype(values.dtype, np.number):
                # create numerical column
                if self.is_backed:
                    self._numerical_array.resize((self.n_index, self.n_columns_num + 1))
                    self._numerical_array[:, -1] = values.astype(self._numerical_array.dtype)

                    self._columns_numerical.resize((self.n_columns_num + 1,))
                    self._columns_numerical[-1] = name

                else:
                    object.__setattr__(self, '_numerical_array',
                                       np.append(self._numerical_array,
                                                 values.astype(self._numerical_array.dtype)[:, None], axis=1))

                    self._columns_numerical.resize((self.n_columns_num + 1,), refcheck=False)
                    self._columns_numerical[-1] = name

            else:
                # create string column
                if self.is_backed:
                    self._string_array.resize((self.n_index, self.n_columns_str + 1))
                    self._string_array[:, -1] = values.astype(str)

                    self._columns_string.resize((self.n_columns_str + 1,))
                    self._columns_string[-1] = name

                else:
                    object.__setattr__(self, '_string_array',
                                       np.append(self._string_array, values.astype(str)[:, None], axis=1))

                    self._columns_string.resize((self.n_columns_str + 1,), refcheck=False)
                    self._columns_string[-1] = name

    @check_can_write
    def __delattr__(self,
                    column_name: str) -> None:
        def drop_column_np(array_: np.ndarray,
                           columns_: np.ndarray,
                           index_: int) -> tuple[np.ndarray, np.ndarray]:
            # delete column from the data array
            array_ = np.delete(array_, index_, 1)

            # delete column from the column names
            columns_ = np.delete(columns_, index_)

            return array_, columns_

        def drop_column_h5(array_: np.ndarray,
                           columns_: np.ndarray,
                           index_: int) -> None:
            # transfer data one row to the left, starting from the column after the one to delete
            # matrix | 0 1 2 3 4 | with index of the column to delete = 2
            #   ==>  | 0 1 3 4 . |
            array_[:, index_:len(columns_) - 1] = array_[:, index_ + 1:len(columns_)]

            # delete column from the column names as above
            columns_[index_:len(columns_) - 1] = columns_[index_ + 1:len(columns_)]

            # resize the arrays to drop the extra column at the end
            columns_.resize((len(columns_) - 1,))
            array_.resize((array_.shape[0], array_.shape[1] - 1))

        if self.has_locked_columns:
            raise VLockError("Cannot delete column from TDF with locked columns.")

        if column_name in self.columns_num:
            item_index = np.where(self.columns_num == column_name)[0][0]

            if self.is_backed:
                drop_column_h5(self._numerical_array, self._columns_numerical, item_index)

            else:
                self._numerical_array, self._columns_numerical = \
                    drop_column_np(self._numerical_array, self._columns_numerical, item_index)

        elif column_name in self.columns_str:
            item_index = np.where(self.columns_str == column_name)[0][0]

            if self.is_backed:
                drop_column_h5(self._string_array, self._columns_string, item_index)

            else:
                self._string_array, self._columns_string = \
                    drop_column_np(self._string_array, self._columns_string, item_index)

        else:
            raise AttributeError(f"'{column_name}' not found in this TemporalDataFrame.")

    @check_can_read
    def __getitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) \
            -> ViewTemporalDataFrame:
        index_slicer, column_num_slicer, column_str_slicer, _ = parse_slicer(self, slicer)

        return ViewTemporalDataFrame(self, index_slicer, column_num_slicer, column_str_slicer)

    def _get_index_positions(self,
                             index_: np.ndarray,
                             repeating_values: bool = False) -> np.ndarray:
        if self._repeating_index:
            if repeating_values:
                index_positions = np.zeros(len(index_), dtype=int)

                index_0 = self.index_at(self.timepoints[0])

                first_positions = npi.indices(index_0, index_[:len(index_0)])
                index_offset = 0

                for tpi in range(self.n_timepoints):
                    index_positions[tpi*len(index_0):(tpi+1)*len(index_0)] = first_positions + index_offset
                    index_offset += len(index_0)

                return index_positions

            index_len_count = 0

            total_index = np.zeros((self.n_timepoints, len(index_)), dtype=int)

            for tpi, tp in enumerate(self.timepoints):
                i_at_tp = self.index_at(tp)
                total_index[tpi] = npi.indices(i_at_tp, index_) + index_len_count

                index_len_count += len(i_at_tp)

            return np.concatenate(total_index)

        return npi.indices(self.index, index_)

    @check_can_write
    def __setitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER],
                    values: Number | np.number | str | Collection | 'TemporalDataFrame' | 'ViewTemporalDataFrame') \
            -> None:
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
        if self.is_backed or index_array is not None:
            if index_array is None:
                index_array = self.index_at(self.timepoints[0]) if self.has_repeating_index else self.index

            index_positions.sort()

            original_positions = self._get_index_positions(index_array)
            values = values[np.argsort(npi.indices(index_positions,
                                                   original_positions[np.isin(original_positions, index_positions)]))]

        if lcn:
            if self.is_backed:
                for column_position, column_name in zip(npi.indices(self.columns_num, column_num_slicer),
                                                        column_num_slicer):
                    self._numerical_array[index_positions, column_position] = \
                        values[:, np.where(columns_array == column_name)[0][0]].astype(float)

            else:
                self.values_num[index_positions[:, None], npi.indices(self._columns_numerical, column_num_slicer)] = \
                    values[:, npi.indices(columns_array, column_num_slicer)].astype(float)

        if lcs:
            values_str = values[:, npi.indices(columns_array, column_str_slicer)].astype(str)
            if values_str.dtype > self._string_array.dtype:
                object.__setattr__(self, '_string_array', self._string_array.astype(values_str.dtype))

            if self.is_backed:
                for column_position, column_name in zip(npi.indices(self.columns_str, column_str_slicer),
                                                        column_str_slicer):
                    self._string_array[index_positions, column_position] = \
                        values_str[:, np.where(columns_array == column_name)[0][0] - lcn]

            else:
                self.values_str[index_positions[:, None], npi.indices(self._columns_string, column_str_slicer)] = \
                    values_str

    def __setstate__(self, state):
        for key, value in state[1].items():
            object.__setattr__(self, key, value)

    @check_can_read
    def __add__(self,
                value: Number | np.number | str | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self._add_core(value)

    @check_can_read
    def __radd__(self, value: Number | np.number | str) -> 'TemporalDataFrame':
        return self.__add__(value)

    @check_can_write
    def __iadd__(self,
                 value: Number | np.number | str | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values incremented by <value> if <value> is a number.
            - <value> appended to string values if <value> is a string.
        """
        if isinstance(value, (Number, np.number)):
            if self.values_num.size == 0:
                raise ValueError("No numerical data to add.")

            self.values_num += value
            return self

        else:
            if self.values_str.size == 0:
                raise ValueError("No string data to add to.")

            # TODO : does not work for backed TDFs, we need to split the class for managing backed TDFs
            if self.is_backed:
                raise NotImplementedError

            self.values_str = np.char.add(self.values_str, value)
            return self

    @check_can_read
    def __sub__(self,
                value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self._op_core(value, 'sub')

    @check_can_read
    def __rsub__(self,
                 value: Number | np.number) -> 'TemporalDataFrame':
        return self.__sub__(value)

    @check_can_write
    def __isub__(self,
                 value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values decremented by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to subtract.")

        self.values_num -= value
        return self

    @check_can_read
    def __mul__(self,
                value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self._op_core(value, 'mul')

    @check_can_read
    def __rmul__(self,
                 value: Number | np.number) -> 'TemporalDataFrame':
        return self.__mul__(value)

    @check_can_write
    def __imul__(self,
                 value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values multiplied by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to multiply.")

        self.values_num *= value
        return self

    @check_can_read
    def __truediv__(self,
                    value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self._op_core(value, 'div')

    @check_can_read
    def __rtruediv__(self,
                     value: Number | np.number) -> 'TemporalDataFrame':
        return self.__truediv__(value)

    @check_can_write
    def __itruediv__(self,
                     value: Number | np.number | 'BaseTemporalDataFrame') -> 'TemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to divide.")

        self.values_num /= value
        return self

    @check_can_read
    def __eq__(self,
               other: Any) -> bool | np.ndarray:
        """
        Test for equality with :
            - another TemporalDataFrame or view of a TemporalDataFrame
            - a single value (either numerical or string)
        """
        return self._is_equal(other)

    @check_can_read
    def __invert__(self) -> ViewTemporalDataFrame:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return ViewTemporalDataFrame(self, np.arange(0, self.n_index), self.columns_num, self.columns_str,
                                     inverted=True)

    def __reload_from_file(self,
                           file: H5Data) -> None:
        """
        Reload data from a H5 file into this TemporalDataFrame. This function is called after a .write() on a TDF that
            was not backed.

        Args:
            file: H5 file to load data from.
        """
        self._name = file.attrs['name']
        self._lock = (file.attrs['locked_indices'], file.attrs['locked_columns'])
        self._timepoints_column_name = None if file.attrs['timepoints_column_name'] == '__TDF_None__' else \
            file.attrs['timepoints_column_name']

        self._index = file['index']
        self._columns_numerical = file['columns_numerical']
        self._columns_string = file['columns_string']
        self._timepoints_array = file['timepoints']
        self._numerical_array = file['values_numerical']
        self._string_array = file['values_string']

        self._file = file

        self._timepoint_masks = dict()

    @property                                                                           # type: ignore
    @check_can_read
    def name(self) -> str:
        return self._name

    @name.setter                                                                        # type: ignore
    @check_can_write
    def name(self,
             name: str) -> None:
        object.__setattr__(self, '_name', str(name))

        if self.is_backed:
            self._file.attrs['name'] = name

    @property                                                                           # type: ignore
    def full_name(self) -> str:
        """
        Get the full name.
        """
        if self.is_backed and not self._file:
            return f"{self.__class__.__name__} backed on closed file"

        parts = []
        if self.empty:
            parts.append('empty')

        if self.is_backed:
            parts.append('backed')

        if len(parts):
            parts[0] = parts[0].capitalize()

        parts += [self.__class__.__name__, self.name]

        return ' '.join(parts)

    @property                                                                           # type: ignore
    @check_can_read
    def file(self) -> H5Data | None:
        """
        Get the file this TemporalDataFrame is backed on.
        It is None if this TemporalDataFrame is not backed on any file.
        """
        return self._file

    @property                                                                           # type: ignore
    @check_can_read
    def is_backed(self) -> bool:
        """
        Whether this TemporalDataFrame is backed on a file or not.
        """
        return self._file is not None

    @property                                                                           # type: ignore
    @check_can_read
    def has_locked_indices(self) -> bool:
        return bool(self._lock[0])

    def lock_indices(self) -> None:
        object.__setattr__(self, '_lock', (True, self.has_locked_columns))

    def unlock_indices(self) -> None:
        object.__setattr__(self, '_lock', (False, self.has_locked_columns))

    @property                                                                           # type: ignore
    @check_can_read
    def has_locked_columns(self) -> bool:
        return bool(self._lock[1])

    def lock_columns(self) -> None:
        object.__setattr__(self, '_lock', (self.has_locked_indices, True))

    def unlock_columns(self) -> None:
        object.__setattr__(self, '_lock', (self.has_locked_indices, False))

    @property                                                                           # type: ignore
    @check_can_read
    def lock(self) -> tuple[bool, bool]:
        return self.has_locked_indices, self.has_locked_columns

    def _empty_numerical(self) -> bool:
        return self._numerical_array.size == 0

    def _empty_string(self) -> bool:
        return self._string_array.size == 0

    @property                                                                           # type: ignore
    @check_can_read
    def empty(self) -> bool:
        """
        Whether this TemporalDataFrame is empty (no numerical data and no string data).
        """
        return self._empty_numerical() and self._empty_string()

    @property                                                                           # type: ignore
    @check_can_read
    def shape(self) -> tuple[int, list[int], int]:
        """
        Get the shape of this TemporalDataFrame as a 3-tuple of :
            - number of time-points
            - number of rows per time-point
            - number of columns
        """
        return self.n_timepoints, \
            [self.n_index_at(tp) for tp in self.timepoints], \
            self.n_columns_num + self.n_columns_str

    def _head_tail(self,
                   n: int) -> str:
        """
        Common function for getting a head or tail representation of this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first/last n rows in this TemporalDataFrame.
        """
        def repr_single_array(array: np.ndarray, columns_: np.ndarray) -> tuple[pd.DataFrame, tuple[int, ...]]:
            tp_data_array_ = array[self.get_timepoint_mask(tp)]

            tp_array_ = np.array([[tp] for _ in range(min(n, tp_data_array_.shape[0]))])

            spacer_ = np.array([['|'] for _ in range(min(n, tp_data_array_.shape[0]))])

            columns_ = np.concatenate(([self._timepoints_column_name, ''], columns_)) if \
                self._timepoints_column_name is not None else np.concatenate(([DEFAULT_TIME_POINTS_COL_NAME, ''],
                                                                              columns_))

            tp_df_ = pd.DataFrame(np.concatenate((tp_array_,
                                                 spacer_,
                                                 tp_data_array_[:n]), axis=1),
                                  index=self.index_at(tp)[:n],
                                  columns=columns_)

            return tp_df_, tp_data_array_.shape

        if not len(timepoints_list := self.timepoints):
            return f"Time points: []\n" \
                   f"Columns: {[col for col in self.columns]}\n" \
                   f"Index: {[idx for idx in self.index]}"

        repr_string = ""

        for tp in timepoints_list[:5]:
            # display the first n rows of the first 5 timepoints in this TemporalDataFrame
            repr_string += f"\033[4mTime point : {repr(tp)}\033[0m\n"

            if not self._empty_numerical() and not self._empty_string():
                tp_mask = self.get_timepoint_mask(tp)

                tp_numerical_array = self._numerical_array[
                    np.repeat(tp_mask[:, None], self.n_columns_num, axis=1)
                ]
                tp_numerical_array = tp_numerical_array.reshape((len(tp_numerical_array) // self.n_columns_num,
                                                                 self.n_columns_num))

                tp_string_array = self.values_str[
                    np.repeat(tp_mask[:, None], self.n_columns_str, axis=1)
                ]
                tp_string_array = tp_string_array.reshape((len(tp_string_array) // self.n_columns_str,
                                                           self.n_columns_str))

                tp_array = np.array([[tp] for _ in range(min(n, tp_numerical_array.shape[0]))])

                spacer = np.array([['|'] for _ in range(min(n, tp_numerical_array.shape[0]))])

                tp_col_name = DEFAULT_TIME_POINTS_COL_NAME if self._timepoints_column_name is None else \
                    self._timepoints_column_name
                columns = np.concatenate(([tp_col_name, ''], self.columns_num, [''], self.columns_str))

                tp_df = pd.DataFrame(np.concatenate((tp_array,
                                                     spacer,
                                                     tp_numerical_array[:n],
                                                     spacer,
                                                     tp_string_array[:n]), axis=1),
                                     index=self.index_at(tp)[:n],
                                     columns=columns)
                tp_shape = (tp_numerical_array.shape[0], tp_numerical_array.shape[1] + tp_string_array.shape[1])

            elif not self._empty_numerical():
                tp_df, tp_shape = repr_single_array(self.values_num, self.columns_num)

            elif not self._empty_string():
                tp_df, tp_shape = repr_single_array(self.values_str, self.columns_str)

            else:
                nb_rows_at_tp = int(np.sum(self.get_timepoint_mask(tp)))

                tp_array_ = np.array([[tp] for _ in range(min(n, nb_rows_at_tp))])

                spacer_ = np.array([['|'] for _ in range(min(n, nb_rows_at_tp))])

                columns_ = [self._timepoints_column_name, ''] if self._timepoints_column_name is not None \
                    else [DEFAULT_TIME_POINTS_COL_NAME, '']

                tp_df = pd.DataFrame(np.concatenate((tp_array_, spacer_), axis=1),
                                     index=self.index_at(tp)[:n],
                                     columns=columns_)

                tp_shape = (tp_df.shape[0], 0)

            # remove unwanted shape display by pandas and replace it by our own
            repr_string += re.sub(r'\\n\[.*$', '', repr(tp_df)) + '\n' + f'[{tp_shape[0]} x {tp_shape[1]}]\n\n'

        # then display only the list of remaining timepoints
        if len(timepoints_list) > 5:
            repr_string += f"\nSkipped time points {repr_array(timepoints_list[5:])} ...\n\n\n"

        return repr_string

    @check_can_read
    def head(self,
             n: int = 5) -> str:
        """
        Get a short representation of the first n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first n rows in this TemporalDataFrame.
        """
        return self._head_tail(n)

    @check_can_read
    def tail(self,
             n: int = 5) -> str:
        """
        Get a short representation of the last n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the last n rows in this TemporalDataFrame.
        """
        # TODO : negative n not handled
        return self._head_tail(-n)

    @property                                                                           # type: ignore
    @check_can_read
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """
        unique_timepoints_idx = np.unique(self._timepoints_array, return_index=True)[1]

        if isinstance(self._timepoints_array, Dataset):
            return np.array([TimePoint(tp.decode()) for tp in self._timepoints_array[sorted(unique_timepoints_idx)]])

        return self._timepoints_array[sorted(unique_timepoints_idx)]

    @property                                                                           # type: ignore
    @check_can_read
    def n_timepoints(self) -> int:
        return len(self.timepoints)

    def __fast_compare(self,
                       comparison_tp: TimePoint | str) -> np.ndarray:
        if not (ltpm := len(self._timepoint_masks)):
            return np.equal(self._timepoints_array, comparison_tp)

        tp_mask = np.zeros(len(self._timepoints_array), dtype=bool)

        if ltpm == 1:
            not_already_computed = ~next(iter(self._timepoint_masks.values()))

        else:
            not_already_computed = np.logical_and.reduce([~mask for mask in self._timepoint_masks.values()])

        tp_mask[not_already_computed] = np.equal(self._timepoints_array[not_already_computed], comparison_tp)
        return tp_mask

    @check_can_read
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
            comparison_tp = str(TimePoint(timepoint)).encode() if isinstance(self._timepoints_array, Dataset) else \
                TimePoint(timepoint)

            self._timepoint_masks[timepoint] = self.__fast_compare(comparison_tp)

        return self._timepoint_masks[timepoint]

    @property                                                                           # type: ignore
    @check_can_read
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """
        if isinstance(self._timepoints_array, Dataset):
            return np.array([TimePoint(tp) for tp in self._timepoints_array.asstr()])

        return self._timepoints_array.copy()

    @property                                                                           # type: ignore
    @check_can_read
    def timepoints_column_str(self) -> np.ndarray:
        """
        Get the column of time-point values cast as strings.
        """
        return np.array(list(map(str, self.timepoints_column)))

    @property
    @check_can_read
    def timepoints_column_numerical(self) -> np.ndarray:
        """
        Get the column of time-point values cast as floats.
        """
        return np.array([tp.value for tp in self.timepoints_column])

    @property                                                                           # type: ignore
    @check_can_read
    def timepoints_column_name(self) -> str | None:
        return self._timepoints_column_name

    @timepoints_column_name.setter
    @check_can_write
    def timepoints_column_name(self, name) -> None:
        object.__setattr__(self, '_timepoints_column_name', str(name))

        if self.is_backed:
            self._file.attrs['timepoints_column_name'] = str(name)

    @property                                                                           # type: ignore
    @check_can_read
    def index(self) -> np.ndarray:
        if isinstance(self._index, Dataset) and self._index.dtype == np.dtype('O'):
            return self._index.asstr()[()]

        return self._index[()].copy()

    def _check_valid_index(self,
                           values: np.ndarray,
                           repeating_index: bool) -> None:
        if not (vs := values.shape) == (s := self._index.shape):
            raise ValueError(f"Shape mismatch, new 'index' values have shape {vs}, expected {s}.")

        if repeating_index:
            first_index = values[self.timepoints_column == self.timepoints[0]]

            for tp in self.timepoints[1:]:
                index_tp = values[self.timepoints_column == tp]

                if not len(first_index) == len(index_tp) or not np.all(first_index == index_tp):
                    raise ValueError(f"Index at time-point {tp} is not equal to index at time-point "
                                     f"{self.timepoints[0]}.")

        else:
            if not self.n_index == len(np.unique(values)):
                raise ValueError("Index values must be all unique.")

    @index.setter                                                                       # type: ignore
    @check_can_write
    def index(self,
              values: np.ndarray) -> None:
        if self.has_locked_indices:
            raise VLockError("Cannot set index in TDF with locked index.")

        self._check_valid_index(values, self._repeating_index)

        self._index[()] = values

    @property
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """
        return self._repeating_index

    @check_can_write
    def set_index(self,
                  values: np.ndarray,
                  repeating_index: bool = False) -> None:
        if self.has_locked_indices:
            raise VLockError("Cannot set index in TDF with locked index.")

        self._check_valid_index(values, repeating_index)

        if self.is_backed and values.dtype != self._index.dtype:
            del self.file['index']
            write_array(values, self.file, 'index')

        else:
            self._index[()] = values

        object.__setattr__(self, '_repeating_index', repeating_index)

    @check_can_write
    def reindex(self,
                values: np.ndarray,
                repeating_index: bool = False) -> None:
        if self.has_locked_indices:
            raise VLockError("Cannot set index in TDF with locked index.")

        # check all values in index
        self._check_valid_index(values, repeating_index)

        if not np.all(np.isin(values, self.index)):
            raise ValueError("New index contains values which are not in the current index.")

        if repeating_index and not self._repeating_index:
            raise ValueError("Cannot set repeating index on TDF with non-repeating index.")

        elif not repeating_index and self._repeating_index:
            raise ValueError("Cannot set non-repeating index on TDF with repeating index.")

        # re-arrange rows to conform to new index
        index_positions = self._get_index_positions(values, repeating_values=True)

        self._numerical_array[:] = self._numerical_array[index_positions]
        self._string_array[:] = self._string_array[index_positions]

        # set index
        object.__setattr__(self, '_index', values)
        object.__setattr__(self, '_repeating_index', repeating_index)

    @property                                                                           # type: ignore
    @check_can_read
    def n_index(self) -> int:
        return len(self.index)

    @check_can_read
    def index_at(self,
                 timepoint: str | TimePoint) -> np.ndarray:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """
        return self.index[self.get_timepoint_mask(timepoint)].copy()

    @check_can_read
    def n_index_at(self,
                   timepoint: str | TimePoint) -> int:
        return len(self.index_at(timepoint))

    @property                                                                           # type: ignore
    @check_can_read
    def columns(self) -> np.ndarray:
        return np.concatenate((self.columns_num, self.columns_str))

    @columns.setter
    @check_can_write
    def columns(self,
                values: np.ndarray):
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in TDF with locked columns.")

        if not (vs := len(values)) == (s := self.n_columns_num + self.n_columns_str):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        self._columns_numerical[()] = values[:self.n_columns_num]
        self._columns_string[()] = values[self.n_columns_num:]

    @check_can_read
    def keys(self) -> np.ndarray:
        return self.columns

    @property                                                                           # type: ignore
    @check_can_read
    def columns_num(self) -> np.ndarray:
        if isinstance(self._columns_numerical, Dataset) and self._columns_numerical.dtype == np.dtype('O'):
            return self._columns_numerical.asstr()[()]

        return self._columns_numerical[()].copy()

    @columns_num.setter
    @check_can_write
    def columns_num(self, values: np.ndarray) -> None:
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in TDF with locked columns.")

        if not (vs := values.shape) == (s := self._columns_numerical.shape):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        self._columns_numerical[()] = values

    @property                                                                           # type: ignore
    @check_can_read
    def n_columns_num(self) -> int:
        return len(self.columns_num)

    @property                                                                           # type: ignore
    @check_can_read
    def columns_str(self) -> np.ndarray:
        if isinstance(self._columns_string, Dataset) and self._columns_string.dtype == np.dtype('O'):
            return self._columns_string.asstr()[()]

        return self._columns_string[()].copy()

    @columns_str.setter
    @check_can_write
    def columns_str(self,
                    values: np.ndarray) -> None:
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in TDF with locked columns.")

        if not (vs := values.shape) == (s := self._columns_string.shape):
            raise ValueError(f"Shape mismatch, new 'columns_str' values have shape {vs}, expected {s}.")

        self._columns_string[()] = values

    @property                                                                           # type: ignore
    @check_can_read
    def n_columns_str(self) -> int:
        return len(self.columns_str)

    @property                                                                           # type: ignore
    @check_can_read
    def values_num(self) -> np.ndarray:
        return self._numerical_array[()]

    @values_num.setter
    @check_can_write
    def values_num(self,
                   values: np.ndarray) -> None:
        self._numerical_array[()] = values

    @property                                                                           # type: ignore
    @check_can_read
    def values_str(self) -> np.ndarray:
        # TODO : avoid loading entire dataset in RAM ??
        #  Maybe create another TDF class specific to backed TDFs
        if isinstance(self._string_array, Dataset):
            return self._string_array.asstr()[()].astype(str)

        return self._string_array

    @values_str.setter
    @check_can_write
    def values_str(self,
                   values: np.ndarray) -> None:
        object.__setattr__(self, '_string_array', self._string_array.astype(values.dtype))
        self._string_array[()] = values

    @check_can_read
    def to_pandas(self,
                  with_timepoints: str | None = None,
                  timepoints_type: Literal['string', 'numerical'] = 'string',
                  str_index: bool = False) -> pd.DataFrame:
        """
        Convert this TemporalDataFrame to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
            timepoints_type: if <with_timepoints> if True, type of the timepoints that will be added (either 'string'
                or 'numerical'). (default: 'string')
            str_index: cast index as string ?
        """
        return self._convert_to_pandas(with_timepoints=with_timepoints,
                                       timepoints_type=timepoints_type,
                                       str_index=str_index)

    @property
    def at(self) -> 'VAtIndexer':
        """
        Access a single value from a pair of row and column labels.
        """
        return VAtIndexer(self)

    @property
    def iat(self) -> 'ViAtIndexer':
        """
        Access a single value from a pair of row and column indices.
        """
        return ViAtIndexer(self)

    @property
    def loc(self) -> 'VLocIndexer':
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
        """
        return VLocIndexer(self)

    @property
    def iloc(self) -> 'ViLocIndexer':
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
        """
        return ViLocIndexer(self)

    @check_can_read
    def write(self,
              file: str | Path | H5Data | None = None) -> None:
        """
        Save this TemporalDataFrame in HDF5 file format.

        Args:
            file: path to save the TemporalDataFrame.
        """
        if file is None:
            if not self.is_backed or not self.file.mode == H5Mode.READ_WRITE:
                raise ValueError("A file path must be supplied when write a TemporalDataFrame that is not "
                                 "already backed on a file.")

            file = self.file

        elif isinstance(file, (str, Path)):
            # open H5 file in 'a' mode: equivalent to r+ and creates the file if it does not exist
            file = File(Path(file), 'a')

        # avoid writing if already backed and writing to this TDF's file.
        # TODO this breaks for relative and ~ paths !
        if not (self.is_backed and self.file.file.filename == file.file.filename):
            write_TDF(self, file)

            self.__reload_from_file(file)

    @check_can_read
    def to_csv(self,
               path: str | Path,
               sep: str = ",",
               na_rep: str = "",
               index: bool = True,
               header: bool = True) -> None:
        """
        Save this TemporalDataFrame in a csv file.

        Args:
            path: a path to the csv file.
            sep: String of length 1. Field delimiter for the output file.
            na_rep: Missing data representation.
            index: Write row names (index) ?
            header: Write out the column names ? If a list of strings is given it is assumed to be aliases for the
                column names.
        """
        tp_col_name = self.timepoints_column_name if self.timepoints_column_name is not None else \
            DEFAULT_TIME_POINTS_COL_NAME

        self.to_pandas(with_timepoints=tp_col_name).to_csv(path, sep=sep, na_rep=na_rep, index=index, header=header)

    @check_can_read
    def copy(self) -> 'TemporalDataFrame':
        """
        Get a copy of this TemporalDataFrame.
        """
        return self._copy()

    @check_can_write
    def insert(self,
               loc: int,
               name: str,
               values: np.ndarray | Iterable | int | float) -> None:
        """
        Insert a column in either the numerical data or the string data, depending on the type of the <values> array.
            The column is inserted at position <loc> with name <name>.
        """
        def insert_column_np(array_: np.ndarray,
                             columns_: np.ndarray,
                             index_: int) -> tuple[np.ndarray, np.ndarray]:
            # insert column in the data array the position index_.
            array_ = np.insert(array_, index_, values, axis=1)

            # insert column in the column names
            columns_ = np.insert(columns_, index_, name)

            return array_, columns_

        def insert_column_h5(array_: np.ndarray,
                             columns_: np.ndarray,
                             index_: int) -> None:
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
            raise VLockError("Cannot insert columns in TDF with locked columns.")

        values = np.array(values)

        if (l := len(values)) != (n := self.n_index):
            raise ValueError(f"Wrong number of values ({l}), expected {n}.")

        if name in self.columns:
            raise ValueError(f"A column named '{name}' already exists.")

        if np.issubdtype(values.dtype, np.number):
            # create numerical column
            if self.is_backed:
                insert_column_h5(self._numerical_array, self._columns_numerical, loc)

            else:
                array, columns = insert_column_np(self._numerical_array, self._columns_numerical, loc)
                object.__setattr__(self, '_numerical_array', array)
                object.__setattr__(self, '_columns_numerical', columns)

        else:
            # create string column
            if self.is_backed:
                insert_column_h5(self._string_array, self._columns_string, loc)

            else:
                array, columns = insert_column_np(self._string_array, self._columns_string, loc)
                object.__setattr__(self, '_string_array', array)
                object.__setattr__(self, '_columns_string', columns)

    @check_can_read
    def merge(self,
              other: TemporalDataFrame | ViewTemporalDataFrame,
              name: str | None = None) -> 'TemporalDataFrame':
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """
        if not np.all(self.timepoints == other.timepoints):
            raise ValueError("Cannot merge TemporalDataFrames with different time points.")

        if not np.all(self.columns_num == other.columns_num) and not np.all(self.columns_str == other.columns_num):
            raise ValueError("Cannot merge TemporalDataFrames with different columns.")

        if not self.timepoints_column_name == other.timepoints_column_name:
            raise ValueError("Cannot merge TemporalDataFrames with different 'timepoints_column_name'.")

        if self.empty:
            combined_index = np.array([])
            for tp in self.time_points:
                combined_index = np.concatenate((combined_index,
                                                 self.index_at(tp).values,
                                                 other.index_at(tp).values))

            _data = pd.DataFrame(index=combined_index, columns=self.columns)

        else:
            _data = None

            for time_point in self.timepoints:
                if np.any(np.isin(other.index_at(time_point), self.index_at(time_point))):
                    raise ValueError(f"TemporalDataFrames to merge have index values in common at time point "
                                     f"'{time_point}'.")

                _data = pd.concat((_data, self[time_point].to_pandas(), other[time_point].to_pandas()))

            _data.columns = _data.columns.astype(self.columns.dtype)

        if self.timepoints_column_name is None:
            _time_list = [time_point for time_point in self.timepoints
                          for _ in range(self.n_index_at(time_point) + other.n_index_at(time_point))]

        else:
            _time_list = None

        return TemporalDataFrame(data=_data,
                                 columns_numerical=self.columns_num,
                                 columns_string=self.columns_str,
                                 time_list=_time_list,
                                 time_col_name=self.timepoints_column_name,
                                 name=name)
