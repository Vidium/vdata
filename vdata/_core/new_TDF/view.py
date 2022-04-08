# coding: utf-8
# Created on 31/03/2022 15:20
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
import numpy_indexed as npi
from pathlib import Path
from numbers import Number
from h5py import Dataset, File

from typing import TYPE_CHECKING, Union, Optional

from vdata.new_time_point import TimePoint
from vdata.utils import repr_array
from .name_utils import H5Mode, H5Data, SLICER
from .base import BaseTemporalDataFrame
from ._write import write_TDF

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame


# ====================================================
# code
def check_can_read(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        # if parent TDF is backed but file is closed : can't read
        if self._parent.is_backed and not self._parent.file:
            raise ValueError("Can't read parent TemporalDataFrame backed on closed file.")

        return func(*args, **kwargs)

    return wrapper


def check_can_write(func):
    def wrapper(*args, **kwargs):
        self = args[0]

        # if parent TDF is backed but file is closed or file mode is not a or r+ : can't write
        if self._parent.is_backed:
            if not self._parent.file:
                raise ValueError("Can't write to parent TemporalDataFrame backed on closed file.")

            if (m := self._parent.file.mode) not in (H5Mode.READ_WRITE_CREATE, H5Mode.READ_WRITE):
                raise ValueError(f"Can't write to parent TemporalDataFrame backed on file with mode='{m}'.")

        return func(*args, **kwargs)

    return wrapper


class ViewTemporalDataFrame(BaseTemporalDataFrame):
    __slots__ = '_parent', '_index', '_columns_numerical', '_columns_string'

    def __init__(self,
                 parent: 'TemporalDataFrame',
                 index: np.ndarray,
                 columns_numerical: np.ndarray,
                 columns_string: np.ndarray):
        self._parent = parent
        self._index = index
        self._columns_numerical = columns_numerical
        self._columns_string = columns_string

    @check_can_read
    def __repr__(self) -> str:
        if self._parent.is_backed and not self._parent.file:
            return "View of backed TemporalDataFrame with closed file."

        if self.empty:
            return f"Empty view of {'backed ' if self._parent.is_backed else ''}TemporalDataFrame '" \
                   f"{self._parent.name}'\n" + self.head()

        return f"View of {'backed ' if self._parent.is_backed else ''}TemporalDataFrame '" \
               f"{self._parent.name}'\n" + self.head()

    @check_can_read
    def __dir__(self):
        return dir(ViewTemporalDataFrame) + list(self.columns)

    @check_can_read
    def __getattr__(self,
                    column_name: str) -> 'ViewTemporalDataFrame':
        """
        Get a single column from this view of a TemporalDataFrame.
        """
        if column_name in self.columns_num:
            return ViewTemporalDataFrame(self._parent, self.index, np.array([column_name]), np.array([]))

        elif column_name in self.columns_str:
            return ViewTemporalDataFrame(self._parent, self.index, np.array([]), np.array([column_name]))

        raise AttributeError(f"'{column_name}' not found in this view of a TemporalDataFrame.")

    @check_can_read
    def __getitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a subset.
        """
        raise NotImplementedError

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
                 value: Union[Number, np.number, str]) -> 'ViewTemporalDataFrame':
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
                 value: Union[Number, np.number]) -> 'ViewTemporalDataFrame':
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
                 value: Union[Number, np.number]) -> 'ViewTemporalDataFrame':
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
                     value: Union[Number, np.number]) -> 'ViewTemporalDataFrame':
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to divide.")

        self.values_num /= value
        return self

    @property
    @check_can_read
    def name(self) -> str:
        return f"view of {self._parent.name}"

    @property
    @check_can_read
    def has_locked_indices(self) -> bool:
        return self._parent.has_locked_indices

    @property
    @check_can_read
    def has_locked_columns(self) -> bool:
        return self._parent.has_locked_columns

    @property
    @check_can_read
    def lock(self) -> tuple[bool, bool]:
        """
        Get the index and columns lock state.
        """
        return self.has_locked_indices, self.has_locked_columns

    @property
    @check_can_read
    def empty(self) -> bool:
        return not self.n_index or not self.n_columns

    def _head_tail(self,
                   n: int) -> str:
        """
        Common function for getting a head or tail representation of this ViewTemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first/last n rows in this ViewTemporalDataFrame.
        """
        def repr_single_array(array: np.ndarray, columns_: np.ndarray) -> tuple[pd.DataFrame, tuple[int, ...]]:
            tp_data_array_ = array[self.get_timepoint_mask(tp)]

            tp_array_ = np.array([[tp] for _ in range(min(n, tp_data_array_.shape[0]))])

            spacer_ = np.array([['|'] for _ in range(min(n, tp_data_array_.shape[0]))])

            columns_ = np.concatenate(([self.timepoints_column_name, ''], columns_)) if \
                self.timepoints_column_name is not None else np.concatenate((['Time-point', ''], columns_))

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
            # display the first n rows of the first 5 timepoints in this ViewTemporalDataFrame
            repr_string += f"\033[4mTime point : {repr(tp)}\033[0m\n"

            if len(self._columns_numerical) and len(self._columns_string):
                tp_mask = self.get_timepoint_mask(tp)

                tp_numerical_array = self.values_num[
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

                tp_col_name = 'Time-point' if self.timepoints_column_name is None else self.timepoints_column_name
                columns = np.concatenate(([tp_col_name, ''], self.columns_num, [''], self.columns_str))

                tp_df = pd.DataFrame(np.concatenate((tp_array,
                                                     spacer,
                                                     tp_numerical_array[:n],
                                                     spacer,
                                                     tp_string_array[:n]), axis=1),
                                     index=self.index_at(tp)[:n],
                                     columns=columns)
                tp_shape = (tp_numerical_array.shape[0], tp_numerical_array.shape[1] + tp_string_array.shape[1])

            elif len(self._columns_numerical):
                tp_df, tp_shape = repr_single_array(self.values_num, self._columns_numerical)

            elif len(self._columns_string):
                tp_df, tp_shape = repr_single_array(self.values_str, self._columns_string)

            else:
                raise ValueError

            repr_string += repr(tp_df) + '\n' + f'[{tp_shape[0]} x {tp_shape[1]}]\n\n'

        # then display only the list of remaining timepoints
        if len(timepoints_list) > 5:
            repr_string += f"\nSkipped time points {repr_array(timepoints_list[5:])} ...\n\n\n"

        return repr_string

    @check_can_read
    def head(self, n: int = 5) -> str:
        """
        Get a short representation of the first n rows in this ViewTemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first n rows in this ViewTemporalDataFrame.
        """
        return self._head_tail(n)

    @check_can_read
    def tail(self, n: int = 5) -> str:
        """
        Get a short representation of the last n rows in this ViewTemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the last n rows in this ViewTemporalDataFrame.
        """
        # TODO : negative n not handled
        return self._head_tail(-n)

    @property
    @check_can_read
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this ViewTemporalDataFrame.
        """
        unique_timepoints = np.unique(self.timepoints_column)

        if isinstance(self.timepoints_column, Dataset):
            return np.array([TimePoint(tp.decode()) for tp in unique_timepoints])

        return unique_timepoints

    @check_can_read
    def get_timepoint_mask(self, timepoint: Union[str, TimePoint]) -> np.ndarray:
        """TODO"""
        if self._parent.is_backed:
            return self._parent.timepoints_column[self.index_positions] == str(TimePoint(timepoint))

        return self._parent.timepoints_column[self.index_positions] == TimePoint(timepoint)

    @property
    @check_can_read
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """
        return self._parent.timepoints_column[self.index_positions]

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
        return self._parent.timepoints_column_name

    @property
    @check_can_read
    def index(self) -> np.ndarray:
        return self._index.copy()

    @property
    @check_can_read
    def n_index(self) -> int:
        return len(self._index)

    @check_can_read
    def index_at(self, timepoint: Union[str, TimePoint]) -> np.ndarray:
        """TODO"""
        return self._parent.index[self.index_positions[self.get_timepoint_mask(timepoint)]]

    @property
    @check_can_read
    def index_positions(self) -> np.ndarray:
        """TODO"""
        indices = []
        cumulated_length = 0

        for tp in self._parent.timepoints:
            pitp = self._parent.index_at(tp)
            indices.append(npi.indices(pitp, self._index[np.in1d(self._index, pitp)]) + cumulated_length)
            cumulated_length += len(pitp)

        return np.concatenate(indices)

    @property
    @check_can_read
    def columns_num(self) -> np.ndarray:
        return self._columns_numerical.copy()

    @property
    @check_can_read
    def n_columns_num(self) -> int:
        return len(self._columns_numerical)

    @property
    @check_can_read
    def columns_num_positions(self) -> np.ndarray:
        return npi.indices(self._parent.columns_num, self._columns_numerical)

    @property
    @check_can_read
    def columns_str(self) -> np.ndarray:
        return self._columns_string.copy()

    @property
    @check_can_read
    def n_columns_str(self) -> int:
        return len(self._columns_string)

    @property
    @check_can_read
    def columns_str_positions(self) -> np.ndarray:
        return npi.indices(self._parent.columns_str, self._columns_string)

    @property
    @check_can_read
    def columns(self) -> np.ndarray:
        return np.concatenate((self._columns_numerical, self._columns_string))

    @property
    @check_can_read
    def n_columns(self) -> int:
        return self.n_columns_num + self.n_columns_str

    @property
    @check_can_read
    def values_num(self) -> np.ndarray:
        return self._parent.values_num[self.index_positions[:, None], self.columns_num_positions]

    @values_num.setter
    @check_can_write
    def values_num(self,
                   values: np.ndarray) -> None:
        if self._parent.is_backed:
            for column_position in self.columns_num_positions:
                self._parent._numerical_array[self.index_positions, column_position] = values[:, column_position]

        else:
            self._parent._numerical_array[self.index_positions[:, None], self.columns_num_positions] = values

    @property
    @check_can_read
    def values_str(self) -> np.ndarray:
        return self._parent.values_str[self.index_positions[:, None], self.columns_str_positions]

    @values_str.setter
    @check_can_write
    def values_str(self,
                   values: np.ndarray) -> None:
        self._parent._string_array = self._parent._string_array.astype(values.dtype)
        self._parent._string_array[self.index_positions[:, None], self.columns_str_positions] = values

    @check_can_read
    def to_pandas(self,
                  with_timepoints: Optional[str] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
        """
        return self._convert_to_pandas(with_timepoints=with_timepoints)

    @check_can_read
    def write(self,
              file: Union[str, Path, H5Data]) -> None:
        """
        Save this view of a TemporalDataFrame in HDF5 file format.

        Args:
            file: path to save the TemporalDataFrame.
        """
        if isinstance(file, (str, Path)):
            # open H5 file in 'a' mode: equivalent to r+ and creates the file if it does not exist
            file = File(Path(file), 'a')

        write_TDF(self, file)

    @check_can_read
    def copy(self) -> 'TemporalDataFrame':
        """
        Get a TemporalDataFrame that is a copy of this view.
        """
        return self._copy()
