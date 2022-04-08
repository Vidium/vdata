# coding: utf-8
# Created on 28/03/2022 11:22
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from h5py import File, Dataset
from pathlib import Path
from numbers import Number

from typing import Union, Optional, Collection, Any, Iterable, Type

from vdata.new_time_point import TimePoint, TimePointRange
from vdata.utils import repr_array
from .name_utils import H5Data, H5Mode, SLICER
from .utils import is_collection
from .base import BaseTemporalDataFrame
from .view import ViewTemporalDataFrame
from ._parse import parse_data
from ._write import write_TDF


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

            if (m := file.mode) not in (H5Mode.READ_WRITE_CREATE, H5Mode.READ_WRITE):
                raise ValueError(f"Can't write to TemporalDataFrame backed on file with mode='{m}'.")

        return func(*args, **kwargs)

    return wrapper


class TemporalDataFrame(BaseTemporalDataFrame):
    """
    An equivalent to pandas DataFrames that includes the notion of time on the rows.
    This class implements a modified sub-setting mechanism to subset on time points, rows and columns
    """

    __slots__ = '_name', '_file', '_numerical_array', '_string_array', '_timepoints_array', '_index', \
                '_columns_numerical', '_columns_string', '_lock', '_timepoints_column_name'

    __attributes = ('name', 'timepoints_column_name', 'index', 'columns_num', 'columns_str', 'values_num', 'values_str')

    def __init__(self,
                 data: Union[None, dict, pd.DataFrame, H5Data] = None,
                 index: Optional[Collection] = None,
                 columns_numerical: Optional[Collection] = None,
                 columns_string: Optional[Collection] = None,
                 time_list: Optional[Collection[Union[Number, str, TimePoint]]] = None,
                 time_col_name: Optional[str] = None,
                 lock: Optional[tuple[bool, bool]] = None,
                 name: Any = 'No_Name'):
        """
        Args:
            data: Optional object containing the data to store in this TemporalDataFrame. It can be :
                - a dictionary of ['column_name': [values]], where [values] has always the same length
                - a pandas DataFrame
                - a H5 File or Group containing numerical and string data.
            index: Optional collection of indices. Must match the total number of rows in this TemporalDataFrame,
                over all time-points.
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
            _lock, _timepoints_column_name, _name = \
            parse_data(data, index, columns_numerical, columns_string, time_list, time_col_name, lock, name)

        object.__setattr__(self, '_file', _file)
        object.__setattr__(self, '_numerical_array', _numerical_array)
        object.__setattr__(self, '_string_array', _string_array)
        object.__setattr__(self, '_timepoints_array', _timepoints_array)
        object.__setattr__(self, '_index', _index)
        object.__setattr__(self, '_columns_numerical', _columns_numerical)
        object.__setattr__(self, '_columns_string', _columns_string)
        object.__setattr__(self, '_lock', _lock)
        object.__setattr__(self, '_timepoints_column_name', _timepoints_column_name)
        object.__setattr__(self, '_name', _name)

        # TODO : implement locking logic

    @check_can_read
    def __repr__(self) -> str:
        if self.is_backed and not self._file:
            return f"Backed TemporalDataFrame on closed file."

        if self.empty:
            return f"Empty {'backed ' if self.is_backed else ''}TemporalDataFrame '{self.name}'\n" + self.head()

        return f"{'Backed ' if self.is_backed else ''}TemporalDataFrame '{self.name}'\n" + self.head()

    @check_can_read
    def __dir__(self) -> Iterable[str]:
        return dir(TemporalDataFrame) + list(self.columns)

    @check_can_read
    def __getattr__(self,
                    column_name: str) -> ViewTemporalDataFrame:
        """
        Get a single column from this TemporalDataFrame.
        """
        if column_name in self.columns_num:
            return ViewTemporalDataFrame(self, self.index, np.array([column_name]), np.array([]))

        elif column_name in self.columns_str:
            return ViewTemporalDataFrame(self, self.index, np.array([]), np.array([column_name]))

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

    def __parse_slicer(self,
                       slicer: tuple[SLICER, SLICER, SLICER]) \
            -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """TODO"""
        def parse_one(axis_slicer: SLICER,
                      cast_type: Union[np.dtype, Type[TimePoint]],
                      range_function: Union[Type[range], Type[TimePointRange]],
                      possible_values: np.ndarray) -> Optional[np.ndarray]:
            if axis_slicer is Ellipsis or (isinstance(axis_slicer, slice) and axis_slicer == slice(None)):
                return None

            elif isinstance(axis_slicer, slice):
                start = possible_values[0] if axis_slicer.start is None else cast_type(axis_slicer.start)
                stop = possible_values[-1] if axis_slicer.stop is None else cast_type(axis_slicer.stop)
                step = cast_type(1, start.unit) if axis_slicer.step is None else cast_type(axis_slicer.step)

                return np.array(list(iter(range_function(start, stop, step))))

            elif isinstance(axis_slicer, range) or is_collection(axis_slicer):
                return np.array(list(map(cast_type, axis_slicer)))

            return np.array([cast_type(axis_slicer)])

        tp_slicer, index_slicer, column_slicer = slicer

        # convert slicers to simple lists of values
        tp_array = parse_one(tp_slicer, TimePoint, TimePointRange, self.timepoints)
        index_array = parse_one(index_slicer, self.index.dtype.type, range, self.index)
        columns_array = parse_one(column_slicer, self.columns.dtype.type, range, self.columns)

        if tp_array is None and index_array is None:
            selected_index = self.index

        elif tp_array is None:
            valid_index = np.in1d(index_array, self.index)

            if not np.all(valid_index):
                raise ValueError(f"Some indices were not found in this TemporalDataFrame "
                                 f"({repr_array(index_array[~valid_index])})")

            uniq, indices = np.unique(np.concatenate([index_array[np.in1d(index_array, self.index_at(tp))]
                                                      for tp in self.timepoints]), return_index=True)
            selected_index = uniq[indices.argsort()]

        elif index_array is None:
            valid_tp = np.in1d(tp_array, self.timepoints)

            if not np.all(valid_tp):
                raise ValueError(f"Some time-points were not found in this TemporalDataFrame "
                                 f"({repr_array(tp_array[~valid_tp])})")

            selected_index = self.index[np.in1d(self.timepoints_column, tp_array)]

        else:
            valid_tp = np.in1d(tp_array, self.timepoints)

            if not np.all(valid_tp):
                raise ValueError(f"Some time-points were not found in this TemporalDataFrame "
                                 f"({repr_array(tp_array[~valid_tp])})")

            valid_index = np.in1d(index_array, self.index)

            if not np.all(valid_index):
                raise ValueError(f"Some indices were not found in this TemporalDataFrame "
                                 f"({repr_array(index_array[~valid_index])})")

            selected_index = np.concatenate([index_array[np.in1d(index_array, self.index_at(tp))]
                                             for tp in tp_array])

        if columns_array is None:
            selected_columns_num = self.columns_num
            selected_columns_str = self.columns_str

        else:
            valid_columns = np.in1d(columns_array, self.columns)

            if not np.all(valid_columns):
                raise ValueError(f"Some columns were not found in this TemporalDataFrame "
                                 f"({repr_array(columns_array[~valid_columns])})")

            selected_columns_num = columns_array[np.in1d(columns_array, self.columns_num)]
            selected_columns_str = columns_array[np.in1d(columns_array, self.columns_str)]

        return selected_index, selected_columns_num, selected_columns_str

    @check_can_read
    def __getitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]]) \
            -> ViewTemporalDataFrame:
        def expand_slicer(s: Union[SLICER,
                                   tuple[SLICER, SLICER],
                                   tuple[SLICER, SLICER, SLICER]]) \
                -> tuple[SLICER, SLICER, SLICER]:
            if isinstance(s, tuple) and len(s) == 3:
                return s

            elif isinstance(s, tuple) and len(s) == 2:
                return s[0], s[1], slice(None)

            elif isinstance(s, tuple) and len(s) == 1:
                return s[0], slice(None), slice(None)

            elif isinstance(s, (Number, np.number, str, TimePoint, range, slice)) \
                or s is Ellipsis \
                    or is_collection(s) and all([isinstance(e, (Number, np.number, str, TimePoint)) for e in s]):
                return s, slice(None), slice(None)

            else:
                raise ValueError("Invalid slicer.")

        index_slicer, column_num_slicer, column_str_slicer = self.__parse_slicer(expand_slicer(slicer))

        return ViewTemporalDataFrame(self, index_slicer, column_num_slicer, column_str_slicer)

    @check_can_read
    def __add__(self,
                value: Union[Number, np.number, str]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self._add_core(value)

    @check_can_write
    def __iadd__(self,
                 value: Union[Number, np.number, str]) -> 'TemporalDataFrame':
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
                value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self._op_core(value, 'sub')

    @check_can_write
    def __isub__(self,
                 value: Union[Number, np.number]) -> 'TemporalDataFrame':
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
                value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self._op_core(value, 'mul')

    @check_can_write
    def __imul__(self,
                 value: Union[Number, np.number]) -> 'TemporalDataFrame':
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
                    value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self._op_core(value, 'div')

    @check_can_write
    def __itruediv__(self,
                     value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to divide.")

        self.values_num /= value
        return self

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
        self._timepoints_column_name = None if file.attrs['time_points_column_name'] == '__TDF_None__' else \
            file.attrs['time_points_column_name']

        self._index = file['index']
        self._columns_numerical = file['columns_numerical']
        self._columns_string = file['columns_string']
        self._timepoints_array = file['timepoints']
        self._numerical_array = file['values_numerical']
        self._string_array = file['values_string']

        self._file = file

    @property
    @check_can_read
    def name(self) -> str:
        return self._name

    @name.setter
    @check_can_write
    def name(self,
             name: str) -> None:
        object.__setattr__(self, '_name', str(name))

        if self.is_backed:
            self._file.attrs['name'] = name

    @property
    @check_can_read
    def file(self) -> Union[None, H5Data]:
        """
        Get the file this TemporalDataFrame is backed on.
        It is None if this TemporalDataFrame is not backed on any file.
        """
        return self._file

    @property
    @check_can_read
    def is_backed(self) -> bool:
        """
        Whether this TemporalDataFrame is backed on a file or not.
        """
        return self._file is not None

    @property
    @check_can_read
    def has_locked_indices(self) -> bool:
        return bool(self._lock[0])

    @property
    @check_can_read
    def has_locked_columns(self) -> bool:
        return bool(self._lock[1])

    @property
    @check_can_read
    def lock(self) -> tuple[bool, bool]:
        return self.has_locked_indices, self.has_locked_columns

    def _empty_numerical(self) -> bool:
        return self._numerical_array.size == 0

    def _empty_string(self) -> bool:
        return self._string_array.size == 0

    @property
    @check_can_read
    def empty(self) -> bool:
        """
        Whether this TemporalDataFrame is empty (no numerical data and no string data).
        """
        return self._empty_numerical() and self._empty_string()

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
                self._timepoints_column_name is not None else np.concatenate((['Time-point', ''], columns_))

            tp_df_ = pd.DataFrame(np.concatenate((tp_array_,
                                                 spacer_,
                                                 tp_data_array_[:n]), axis=1),
                                  index=self.index_at(tp)[:n],
                                  columns=columns_)

            return tp_df_, tp_data_array_.shape

        if not len(timepoints_list := self.timepoints):
            return f"Time points: []\n" \
                   f"Columns: {[col for col in self.columns]}\n" \
                   f"Index: {[idx for idx in self._index]}"

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

                tp_col_name = 'Time-point' if self._timepoints_column_name is None else self._timepoints_column_name
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
                raise ValueError

            repr_string += repr(tp_df) + '\n' + f'[{tp_shape[0]} x {tp_shape[1]}]\n\n'

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

    @property
    @check_can_read
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """
        unique_timepoints = np.unique(self._timepoints_array)

        if isinstance(self._timepoints_array, Dataset):
            return np.array([TimePoint(tp.decode()) for tp in unique_timepoints])

        return unique_timepoints

    @check_can_read
    def get_timepoint_mask(self,
                           timepoint: Union[str, TimePoint]) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """
        if isinstance(self._timepoints_array, Dataset):
            return np.equal(self._timepoints_array, str(TimePoint(timepoint)).encode())

        return self._timepoints_array == TimePoint(timepoint)

    @property
    @check_can_read
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """
        if isinstance(self._timepoints_array, Dataset):
            return self._timepoints_array.asstr()[()]

        return self._timepoints_array.copy()

    @property
    @check_can_read
    def timepoints_column_str(self) -> np.ndarray:
        """
        Get the column of time-point values cast as strings.
        """
        return np.array(list(map(str, self.timepoints_column)))

    @property
    @check_can_read
    def timepoints_column_name(self) -> Optional[str]:
        return self._timepoints_column_name

    @timepoints_column_name.setter
    @check_can_write
    def timepoints_column_name(self, name) -> None:
        object.__setattr__(self, '_timepoints_column_name', str(name))

        if self.is_backed:
            self._file.attrs['time_points_column_name'] = str(name)

    @property
    @check_can_read
    def index(self) -> np.ndarray:
        if isinstance(self._index, Dataset) and self._index.dtype == np.dtype('O'):
            return self._index.asstr()[()]

        return self._index[()].copy()

    @index.setter
    @check_can_write
    def index(self,
              values: np.ndarray) -> None:
        if not (vs := values.shape) == (s := self._index.shape):
            raise ValueError(f"Shape mismatch, new 'index' values have shape {vs}, expected {s}.")

        object.__setattr__(self, '_index', values)

    @property
    @check_can_read
    def n_index(self) -> int:
        return len(self.index)

    @check_can_read
    def index_at(self,
                 timepoint: Union[str, TimePoint]) -> np.ndarray:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """
        return self._index[self.get_timepoint_mask(timepoint)].copy()

    @property
    @check_can_read
    def columns(self) -> np.ndarray:
        return np.concatenate((self.columns_num, self.columns_str))

    @property
    @check_can_read
    def columns_num(self) -> np.ndarray:
        if isinstance(self._columns_numerical, Dataset) and self._columns_numerical.dtype == np.dtype('O'):
            return self._columns_numerical.asstr()[()]

        return self._columns_numerical[()].copy()

    @columns_num.setter
    @check_can_write
    def columns_num(self, values: np.ndarray) -> None:
        if not (vs := values.shape) == (s := self._columns_numerical.shape):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        object.__setattr__(self, '_columns_numerical', values)

    @property
    @check_can_read
    def n_columns_num(self) -> int:
        return len(self.columns_num)

    @property
    @check_can_read
    def columns_str(self) -> np.ndarray:
        if isinstance(self._columns_string, Dataset) and self._columns_string.dtype == np.dtype('O'):
            return self._columns_string.asstr()[()]

        return self._columns_string[()].copy()

    @columns_str.setter
    @check_can_write
    def columns_str(self,
                    values: np.ndarray) -> None:
        if not (vs := values.shape) == (s := self._columns_string.shape):
            raise ValueError(f"Shape mismatch, new 'columns_str' values have shape {vs}, expected {s}.")

        object.__setattr__(self, '_columns_string', values)

    @property
    @check_can_read
    def n_columns_str(self) -> int:
        return len(self.columns_str)

    @property
    @check_can_read
    def values_num(self) -> np.ndarray:
        return self._numerical_array[()]

    @values_num.setter
    @check_can_write
    def values_num(self,
                   values: np.ndarray) -> None:
        self._numerical_array[()] = values

    @property
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
                  with_timepoints: Optional[str] = None) -> pd.DataFrame:
        """
        Convert this TemporalDataFrame to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
        """
        return self._convert_to_pandas(with_timepoints=with_timepoints)

    @check_can_read
    def write(self,
              file: Optional[Union[str, Path, H5Data]] = None) -> None:
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
        if not (self.is_backed and self.file.filename == file.filename):  # TODO this breaks for relative and ~ paths !
            write_TDF(self, file)

            self.__reload_from_file(file)

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
               values: np.ndarray) -> None:
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
