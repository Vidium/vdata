# coding: utf-8
# Created on 06/04/2022 11:32
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import numpy_indexed as npi
from pathlib import Path
from numbers import Number
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Any, Collection, Literal, Iterable, Type, NoReturn
from typing_extensions import Self

from vdata.h5pickle import File

from vdata.core.tdf import dataframe, indexer
from vdata.core.tdf.utils import parse_slicer, parse_values
from vdata.core.tdf.name_utils import SLICER, H5Data, DEFAULT_TIME_POINTS_COL_NAME
from vdata.utils import are_equal, repr_array
from vdata.time_point import TimePoint, mean as tp_mean
from vdata.IO import VLockError
from vdata.h5pickle.name_utils import H5Mode

if TYPE_CHECKING:
    from vdata.core.tdf.dataframe import TemporalDataFrame
    from vdata.core.tdf.indexer import VAtIndexer, ViAtIndexer, VLocIndexer, ViLocIndexer
    from vdata.core.dataset_proxy import DatasetProxy


# ====================================================
# code
class BaseTemporalDataFrame(ABC):
    """
    Abstract base class for all TemporalDataFrames.
    """

    # region magic methods
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    @abstractmethod
    def __dir__(self) -> Iterable[str]:
        pass

    @abstractmethod
    def __getattr__(self,
                    column_name: str) -> BaseTemporalDataFrameView:
        """
        Get a single column.
        """

    @abstractmethod
    def __getitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) \
            -> BaseTemporalDataFrameView:
        """
        Get a subset.
        """

    @abstractmethod
    def __setitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER],
                    values: Number | np.number | str | Collection | BaseTemporalDataFrame) -> None:
        """
        Set values in a subset.
        """

    def _check_compatibility(self,
                             value: BaseTemporalDataFrame) -> None:
        # time-points column and nb of columns must be identical
        if not np.array_equal(self.timepoints_column, value.timepoints_column):
            raise ValueError("Time-points do not match.")
        if not np.array_equal(self.n_columns_num, value.n_columns_num):
            raise ValueError("Columns numerical do not match.")
        if not np.array_equal(self.n_columns_str, value.n_columns_str):
            raise ValueError("Columns string do not match.")

    def _add_core(self,
                  value: Number | np.number | str | BaseTemporalDataFrame) -> TemporalDataFrame:
        """
        Internal function for adding a value, called from __add__. Do not use directly.
        """
        if isinstance(value, (Number, np.number)):
            if self.values_num.size == 0:
                raise ValueError("No numerical data to add to.")

            values_num = self.values_num + value
            values_str = self.values_str
            value_name = value

        elif isinstance(value, BaseTemporalDataFrame):
            self._check_compatibility(value)

            values_num = self.values_num + value.values_num
            values_str = np.char.add(self.values_str, value.values_str)
            value_name = value.full_name

        else:
            if self.values_str.size == 0:
                raise ValueError("No string data to add to.")

            values_num = self.values_num
            values_str = np.char.add(self.values_str, value)
            value_name = value

        if self.timepoints_column_name is None:
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index[:], columns=self.columns_num[:]),
                                 pd.DataFrame(values_str, index=self.index[:], columns=self.columns_str[:])),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_list=self.timepoints_column,
                                               lock=self.lock,
                                               name=f"{self.full_name} + {value_name}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index[:], columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index[:], columns=self.columns_num[:]),
                                 pd.DataFrame(values_str, index=self.index[:], columns=self.columns_str[:])),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_col_name=self.timepoints_column_name,
                                               lock=self.lock,
                                               name=f"{self.full_name} + {value_name}")

    def __add__(self,
                value: Number | np.number | str | BaseTemporalDataFrame) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self._add_core(value)

    def __radd__(self,
                 value: Number | np.number | str) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """
        return self.__add__(value)

    def _iadd_str(self,
                  value: str) -> Self:
        """Inplace modification of the string values."""
        self.values_str = np.char.add(self.values_str, value)
        return self

    def __iadd__(self,
                 value: Number | np.number | str | BaseTemporalDataFrame) -> Self:
        """
        Modify inplace the values :
            - numerical values incremented by <value> if <value> is a number.
            - <value> appended to string values if <value> is a string.
        """
        if isinstance(value, (Number, np.number)):
            if self.values_num.size == 0:
                raise ValueError("No numerical data to add to.")

            self.values_num += value
            return self

        else:
            if self.values_str.size == 0:
                raise ValueError("No string data to add to.")

            return self._iadd_str(value)

    def _op_core(self,
                 value: Number | np.number | BaseTemporalDataFrame,
                 operation: Literal['sub', 'mul', 'div']) -> TemporalDataFrame:
        """
        Internal function for subtracting, multiplying by and dividing by a value, called from __add__. Do not use
        directly.
        """
        if operation == 'sub':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to subtract.")
            op = '-'

            if isinstance(value, BaseTemporalDataFrame):
                self._check_compatibility(value)

                values_num = self.values_num - value.values_num
                value_name = value.full_name

            else:
                values_num = self.values_num - value
                value_name = value

        elif operation == 'mul':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to multiply.")
            op = '*'

            if isinstance(value, BaseTemporalDataFrame):
                self._check_compatibility(value)

                values_num = self.values_num * value.values_num
                value_name = value.full_name

            else:
                values_num = self.values_num * value
                value_name = value

        elif operation == 'div':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to divide.")
            op = '/'

            if isinstance(value, BaseTemporalDataFrame):
                self._check_compatibility(value)

                values_num = self.values_num / value.values_num
                value_name = value.full_name

            else:
                values_num = self.values_num / value
                value_name = value

        else:
            raise ValueError(f"Unknown operation '{operation}'.")

        if self.timepoints_column_name is None:
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index[:], columns=self.columns_num[:]),
                                 pd.DataFrame(self.values_str, index=self.index[:], columns=self.columns_str[:])),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_list=self.timepoints_column,
                                               lock=self.lock,
                                               name=f"{self.full_name} {op} {value_name}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index[:], columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index[:], columns=self.columns_num[:]),
                                 pd.DataFrame(self.values_str, index=self.index[:], columns=self.columns_str[:])),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_col_name=self.timepoints_column_name,
                                               lock=self.lock,
                                               name=f"{self.full_name} {op} {value_name}")

    def __sub__(self,
                value: Number | np.number | BaseTemporalDataFrame) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self._op_core(value, 'sub')

    def __rsub__(self,
                 value: Number | np.number) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """
        return self.__sub__(value)

    def __isub__(self,
                 value: Number | np.number | BaseTemporalDataFrame) -> Self:
        """
        Modify inplace the values :
            - numerical values decremented by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to subtract.")

        self.values_num -= value
        return self

    def __mul__(self,
                value: Number | np.number | BaseTemporalDataFrame) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self._op_core(value, 'mul')

    def __rmul__(self,
                 value: Number | np.number) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """
        return self.__mul__(value)

    def __imul__(self,
                 value: Number | np.number | BaseTemporalDataFrame) -> Self:
        """
        Modify inplace the values :
            - numerical values multiplied by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to multiply.")

        self.values_num *= value
        return self

    def __truediv__(self,
                    value: Number | np.number | BaseTemporalDataFrame) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self._op_core(value, 'div')

    def __rtruediv__(self,
                     value: Number | np.number) -> TemporalDataFrame:
        """
        Get a copy with :
            - numerical values divided by <value>.
        """
        return self.__truediv__(value)

    def __itruediv__(self,
                     value: Number | np.number | BaseTemporalDataFrame) -> Self:
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """
        if self.values_num.size == 0:
            raise ValueError("No numerical data to divide.")

        self.values_num /= value
        return self

    def __eq__(self,
               other: Any) -> bool | np.ndarray:
        """
        Test for equality with :
            - another TemporalDataFrame or view of a TemporalDataFrame
            - a single value (either numerical or string)
        """
        if isinstance(other, BaseTemporalDataFrame):
            for attr in ['timepoints_column_name', 'has_locked_indices', 'has_locked_columns', 'columns',
                         'timepoints_column', 'index', 'values_num', 'values_str']:
                if not are_equal(getattr(self, attr), getattr(other, attr)):
                    print(attr)
                    return False

            return True

        if isinstance(other, (Number, np.number)):
            return self.values_num == other

        elif isinstance(other, str):
            return self.values_str == other

        raise ValueError(f"Cannot compare {self.__class__.__name__} object with object of class "
                         f"{other.__class__.__name__}.")

    @abstractmethod
    def __invert__(self) -> BaseTemporalDataFrameView:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """

    # endregion

    # region attributes
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name.
        """

    @property
    @abstractmethod
    def full_name(self) -> str:
        """
        Get the full name.
        """

    @property
    def lock(self) -> tuple[bool, bool]:
        """
        Get the index and columns lock state.
        """
        return self.has_locked_indices, self.has_locked_columns

    @property
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

    @property
    @abstractmethod
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """

    @property
    def n_timepoints(self) -> int:
        return len(self.timepoints)

    @property
    @abstractmethod
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """

    @property
    def timepoints_column_str(self) -> np.ndarray:
        """
        Get the column of time-point values cast as strings.
        """
        return np.array(list(map(str, self.timepoints_column)))

    @property
    def timepoints_column_numerical(self) -> np.ndarray:
        """
        Get the column of time-point values cast as floats.
        """
        return np.array([tp.value for tp in self.timepoints_column])

    @property
    @abstractmethod
    def timepoints_column_name(self) -> str | None:
        """
        Get the name of the column containing the time-points values.
        """

    @property
    @abstractmethod
    def index(self) -> np.ndarray | DatasetProxy:
        """
        Get the index across all time-points.
        """

    @property
    @abstractmethod
    def n_index(self) -> int:
        """
        Get the length of the index.
        """

    @property
    @abstractmethod
    def columns_num(self) -> np.ndarray | DatasetProxy:
        """
        Get the list of column names for numerical data.
        """

    @property
    @abstractmethod
    def n_columns_num(self) -> int:
        """
        Get the number of numerical data columns.
        """

    @property
    @abstractmethod
    def columns_str(self) -> np.ndarray | DatasetProxy:
        """
        Get the list of column names for string data.
        """

    @property
    @abstractmethod
    def n_columns_str(self) -> int:
        """
        Get the number of string data columns.
        """

    @property
    def columns(self) -> np.ndarray:
        """
        Get the list of all column names.
        """
        return np.concatenate((self.columns_num[:], self.columns_str[:]))

    @property
    def n_columns(self) -> int:
        return self.n_columns_num + self.n_columns_str

    @property
    @abstractmethod
    def values_num(self) -> np.ndarray:
        """
        Get the numerical data.
        """

    @values_num.setter
    @abstractmethod
    def values_num(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the numerical data.
        """

    @property
    @abstractmethod
    def values_str(self) -> np.ndarray:
        """
        Get the string data.
        """

    @values_str.setter
    @abstractmethod
    def values_str(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the string data.
        """

    @property
    def values(self) -> np.ndarray:
        """
        Get all the data (num and str concatenated).
        """
        return np.hstack((self.values_num.astype(object), self.values_str.astype(object)))

    @property
    def tp0(self) -> TimePoint:
        return self.timepoints[0]

    @property
    def at(self) -> VAtIndexer:
        """
        Access a single value from a pair of row and column labels.
        """
        return indexer.VAtIndexer(self)

    @property
    def iat(self) -> ViAtIndexer:
        """
        Access a single value from a pair of row and column indices.
        """
        return indexer.ViAtIndexer(self)

    @property
    def loc(self) -> VLocIndexer:
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
        return indexer.VLocIndexer(self)

    @property
    def iloc(self) -> ViLocIndexer:
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
        return indexer.ViLocIndexer(self)

    # endregion

    # region predicates
    @property
    @abstractmethod
    def has_locked_indices(self) -> bool:
        """
        Is the "index" axis locked for modification ?
        """

    @property
    @abstractmethod
    def has_locked_columns(self) -> bool:
        """
        Is the "columns" axis locked for modification ?
        """

    @property
    @abstractmethod
    def empty(self) -> bool:
        """
        Is there data ?
        """

    @property
    @abstractmethod
    def is_view(self) -> bool:
        """
        Is this a view on a TemporalDataFrame ?
        """

    @property
    def is_backed(self) -> bool:
        """
        Is this TemporalDataFrame backed on a file ?
        """
        return False

    # endregion

    # region methods
    @abstractmethod
    def _head_tail(self,
                   n: int) -> str:
        """
        Common function for getting a head or tail representation of this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first/last n rows in this TemporalDataFrame.
        """

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

    @abstractmethod
    def get_timepoint_mask(self,
                           timepoint: str | TimePoint) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """

    @abstractmethod
    def index_at(self,
                 timepoint: str | TimePoint) -> np.ndarray:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """

    def n_index_at(self,
                   timepoint: str | TimePoint) -> int:
        """
        Get the number of rows existing at the given time-point.
        """
        return len(self.index_at(timepoint))

    def _min_max_mean_core(self,
                           axis: int | None,
                           func: Literal['min', 'max', 'mean']) -> float | TemporalDataFrame:
        if axis is None:
            return getattr(np, func)(self.values_num)

        elif axis == 0:
            # only valid if index is the same at all time-points
            i0 = self.index_at(self.tp0)
            for tp in self.timepoints[1:]:
                if not np.array_equal(i0, self.index_at(tp)):
                    raise ValueError(f"Can't take '{func}' along axis 0 if indices are not the same at all "
                                     f"time-points.")

            mmm_tp = {'min': min,
                      'max': max,
                      'mean': tp_mean}[func](self.timepoints)

            return dataframe.TemporalDataFrame(data=pd.DataFrame(
                getattr(np, func)([self.values_num[self.get_timepoint_mask(tp)] for tp in self.timepoints], axis=0),
                index=i0,
                columns=self.columns_num[:],
            ),
                time_list=[mmm_tp for _ in enumerate(i0)],
                time_col_name=self.timepoints_column_name)

        elif axis == 1:
            return dataframe.TemporalDataFrame(data=pd.DataFrame(
                [getattr(np, func)(self.values_num[self.get_timepoint_mask(tp)], axis=0) for tp in self.timepoints],
                index=[func for _ in enumerate(self.timepoints)],
                columns=self.columns_num[:]
            ),
                repeating_index=True,
                time_list=self.timepoints,
                time_col_name=self.timepoints_column_name)

        elif axis == 2:
            return dataframe.TemporalDataFrame(data=pd.DataFrame(
                getattr(np, func)(self.values_num, axis=1),
                index=self.index,
                columns=[func],
            ),
                time_list=self.timepoints_column,
                time_col_name=self.timepoints_column_name)

        raise ValueError(f"Invalid axis '{axis}', should be in [0, 1, 2].")

    def min(self,
            axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the min value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global min). (default: None)
        """
        return self._min_max_mean_core(axis, 'min')

    def max(self,
            axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the max value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global max). (default: None)
        """
        return self._min_max_mean_core(axis, 'max')

    def mean(self,
             axis: int | None = None) -> float | TemporalDataFrame:
        """
        Get the mean value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global mean). (default: None)
        """
        return self._min_max_mean_core(axis, 'mean')

    # endregion

    # region data methods
    def _convert_to_pandas(self,
                           with_timepoints: str | None = None,
                           timepoints_type: Literal['string', 'numerical'] = 'string',
                           str_index: bool = False) -> pd.DataFrame:
        """
        Internal function for converting to a pandas DataFrame. Do not use directly, it is called by '.to_pandas()'.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
            timepoints_type: if <with_timepoints> if True, type of the timepoints that will be added (either 'string'
                or 'numerical'). (default: 'string')
            str_index: cast index as string ?
        """
        index_ = self.index[:]
        if str_index:
            index_ = index_.astype(str)

        if with_timepoints is None:
            return pd.concat((pd.DataFrame(self.values_num if self.values_num.size else None,
                                           index=index_, columns=self.columns_num),
                              pd.DataFrame(self.values_str if self.values_str.size else None,
                                           index=index_, columns=self.columns_str[:])),
                             axis=1)

        if timepoints_type == 'string':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_str[:, None], index=index_, columns=[str(with_timepoints)]),
                pd.DataFrame(self.values_num if self.values_num.size else None,
                             index=index_, columns=self.columns_num[:]),
                pd.DataFrame(self.values_str if self.values_str.size else None,
                             index=index_, columns=self.columns_str[:])
            ), axis=1)

        elif timepoints_type == 'numerical':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_numerical[:, None], index=index_, columns=[str(with_timepoints)]),
                pd.DataFrame(self.values_num if self.values_num.size else None,
                             index=index_, columns=self.columns_num[:]),
                pd.DataFrame(self.values_str if self.values_num.size else None,
                             index=index_, columns=self.columns_str[:])
            ), axis=1)

        raise ValueError(f"Invalid timepoints_type argument '{timepoints_type}'. Should be 'string' or 'numerical'.")

    def to_pandas(self,
                  with_timepoints: str | None = None,
                  timepoints_type: Literal['string', 'numerical'] = 'string',
                  str_index: bool = False) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

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

    def write(self,
              file: str | Path | H5Data | None = None) -> None:
        """
        Save in HDF5 file format.

        Args:
            file: path to save the data.
        """
        if file is None:
            if self.is_view or not self.is_backed or not self._file.mode == H5Mode.READ_WRITE:
                raise ValueError("A file path must be supplied when writing a TemporalDataFrame that is not "
                                 "already backed on a file or when writing a view.")

            file = self._file

        elif isinstance(file, (str, Path)):
            # open H5 file in 'a' mode: equivalent to r+ and creates the file if it does not exist
            file = File(Path(file), 'a')

        # avoid writing if already backed and writing to this tdf's file.
        # TODO this breaks for relative and ~ paths !
        if self.is_view or not (self.is_backed and self._file.filename == file.file.filename):
            # TODO : refactor
            from vdata.read_write.write import write_TDF
            write_TDF(self, file)

    def to_csv(self, path: str | Path, sep: str = ",", na_rep: str = "",
               index: bool = True, header: bool = True) -> None:
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

    def copy(self) -> TemporalDataFrame:
        """
        Get a copy.
        """
        if self.timepoints_column_name is None:
            return dataframe.TemporalDataFrame(self.to_pandas(),
                                               repeating_index=self.has_repeating_index,
                                               time_list=self.timepoints_column,
                                               lock=self.lock,
                                               name=f"copy of {self.name}")

        return dataframe.TemporalDataFrame(self.to_pandas(with_timepoints=self.timepoints_column_name),
                                           repeating_index=self.has_repeating_index,
                                           time_col_name=self.timepoints_column_name,
                                           lock=self.lock,
                                           name=f"copy of {self.name}")

    @abstractmethod
    def merge(self,
              other: BaseTemporalDataFrame,
              name: str | None = None) -> TemporalDataFrame:
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """

    # endregion


class BaseTemporalDataFrameImplementation(BaseTemporalDataFrame, ABC):
    """
    Abstract base class for non-views of TemporalDataFrames.
    """
    _attributes = ('name', 'index', 'columns', 'columns_num', 'columns_str', 'values_num', 'values_str',
                   '_numerical_array', '_string_array', '_index', '_columns_numerical',
                   '_columns_string', '_timepoints_array')

    # region magic methods
    def __repr__(self) -> str:
        return f"{self.full_name}\n" \
               f"{self.head()}"

    def __dir__(self) -> Iterable[str]:
        return dir(BaseTemporalDataFrame) + list(map(str, self.columns))

    def _get_index_positions(self,
                             index_: np.ndarray,
                             repeating_values: bool = False) -> np.ndarray:
        if self.has_repeating_index:
            if repeating_values:
                index_positions = np.zeros(len(index_), dtype=int)

                index_0 = self.index_at(self.timepoints[0])

                first_positions = npi.indices(index_0, index_[:len(index_0)])
                index_offset = 0

                for tpi in range(self.n_timepoints):
                    index_positions[tpi * len(index_0):(tpi + 1) * len(index_0)] = first_positions + index_offset
                    index_offset += len(index_0)

                return index_positions

            index_len_count = 0

            total_index = np.zeros((self.n_timepoints, len(index_)), dtype=int)

            for tpi, tp in enumerate(self.timepoints):
                i_at_tp = self.index_at(tp)
                total_index[tpi] = npi.indices(i_at_tp, index_) + index_len_count

                index_len_count += len(i_at_tp)

            return np.concatenate(total_index)

        return npi.indices(self.index[:], index_)

    # endregion

    # region attributes
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name.
        """

    @name.setter
    @abstractmethod
    def name(self,
             name: str) -> None:
        """Set the name."""

    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        parts = []
        if self.empty:
            parts.append('empty')

        if self.is_backed:
            parts.append('backed')

        if len(parts):
            parts[0] = parts[0].capitalize()

        parts += ['TemporalDataFrame', self.name]

        return ' '.join(parts)

    @property
    @abstractmethod
    def index(self) -> np.ndarray | DatasetProxy:
        """
        Get the index across all time-points.
        """

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

    @index.setter
    def index(self,
              values: np.ndarray) -> None:
        """
        Set the index for rows across all time-points.
        """
        if self.has_locked_indices:
            raise VLockError("Cannot set index in tdf with locked index.")

        self._check_valid_index(values, self._repeating_index)

        self.index[:] = values

    @property
    def n_index(self) -> int:
        return len(self.index)

    @property
    @abstractmethod
    def columns_num(self) -> np.ndarray | DatasetProxy:
        """
        Get the list of column names for numerical data.
        """

    @columns_num.setter
    @abstractmethod
    def columns_num(self,
                    values: np.ndarray) -> None:
        """
        Set the list of column names for numerical data.
        """

    @property
    def n_columns_num(self) -> int:
        """
        Get the number of numerical data columns.
        """
        return len(self.columns_num)

    @property
    @abstractmethod
    def columns_str(self) -> np.ndarray | DatasetProxy:
        """
        Get the list of column names for string data.
        """

    @columns_str.setter
    @abstractmethod
    def columns_str(self,
                    values: np.ndarray) -> None:
        """
        Set the list of column names for string data.
        """

    @property
    def n_columns_str(self) -> int:
        """
        Get the number of string data columns.
        """
        return len(self.columns_str)

    @property
    def columns(self) -> np.ndarray:
        """
        Get the list of all column names.
        """
        return super().columns

    @columns.setter
    def columns(self,
                values: np.ndarray):
        """
        Set the list of all column names.
        """
        if self.has_locked_columns:
            raise VLockError("Cannot set columns in tdf with locked columns.")

        if not (vs := len(values)) == (s := self.n_columns_num + self.n_columns_str):
            raise ValueError(f"Shape mismatch, new 'columns_num' values have shape {vs}, expected {s}.")

        self.columns_num[:] = values[:self.n_columns_num]
        self.columns_str[:] = values[self.n_columns_num:]

    # endregion

    # region predicates
    def _empty_numerical(self) -> bool:
        return self.values_num.size == 0

    def _empty_string(self) -> bool:
        return self.values_str.size == 0

    @property
    def empty(self) -> bool:
        """
        Whether this TemporalDataFrame is empty (no numerical data and no string data).
        """
        return self._empty_numerical() and self._empty_string()

    @property
    def is_view(self) -> bool:
        """
        Is this a view on a TemporalDataFrame ?
        """
        return False

    @property
    @abstractmethod
    def has_repeating_index(self) -> bool:
        """
        Is the index repeated at each time-point ?
        """

    # endregion

    # region methods
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

            columns_ = np.concatenate(([self.timepoints_column_name, ''], columns_)) if \
                self.timepoints_column_name is not None else np.concatenate(([DEFAULT_TIME_POINTS_COL_NAME, ''],
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

                tp_numerical_array = self.values_num[tp_mask]
                tp_string_array = self.values_str[tp_mask]

                tp_array = np.array([[tp] for _ in range(min(n, tp_numerical_array.shape[0]))])

                spacer = np.array([['|'] for _ in range(min(n, tp_numerical_array.shape[0]))])

                tp_col_name = DEFAULT_TIME_POINTS_COL_NAME if self.timepoints_column_name is None else \
                    self._timepoints_column_name
                columns = np.concatenate(([tp_col_name, ''], self.columns_num[:], [''], self.columns_str[:]))

                tp_df = pd.DataFrame(np.concatenate((tp_array,
                                                     spacer,
                                                     tp_numerical_array[:n],
                                                     spacer,
                                                     tp_string_array[:n]), axis=1),
                                     index=self.index_at(tp)[:n],
                                     columns=columns)
                tp_shape = (tp_numerical_array.shape[0], tp_numerical_array.shape[1] + tp_string_array.shape[1])

            elif not self._empty_numerical():
                tp_df, tp_shape = repr_single_array(self.values_num, self.columns_num[:])

            elif not self._empty_string():
                tp_df, tp_shape = repr_single_array(self.values_str, self.columns_str[:])

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

    @abstractmethod
    def lock_indices(self) -> None:
        """Lock the "index" axis to prevent modifications."""

    @abstractmethod
    def unlock_indices(self) -> None:
        """Unlock the "index" axis to allow modifications."""

    @abstractmethod
    def lock_columns(self) -> None:
        """Lock the "columns" axis to prevent modifications."""

    @abstractmethod
    def unlock_columns(self) -> None:
        """Unlock the "columns" axis to allow modifications."""

    @abstractmethod
    def set_index(self,
                  values: np.ndarray,
                  repeating_index: bool = False) -> None:
        """Set new index values."""

    def reindex(self,
                order: np.ndarray,
                repeating_index: bool = False) -> None:
        """Re-order rows in this TemporalDataFrame so that their index matches the new given order."""
        if self.has_locked_indices:
            raise VLockError("Cannot set index in tdf with locked index.")

        # check all values in index
        self._check_valid_index(order, repeating_index)

        if not np.all(np.isin(order, self.index)):
            raise ValueError("New index contains values which are not in the current index.")

        if repeating_index and not self.has_repeating_index:
            raise ValueError("Cannot set repeating index on tdf with non-repeating index.")

        elif not repeating_index and self.has_repeating_index:
            raise ValueError("Cannot set non-repeating index on tdf with repeating index.")

        # re-arrange rows to conform to new index
        index_positions = self._get_index_positions(order, repeating_values=True)

        # use `[:]` before actually indexing to cast h5 Datasets to numpy since we cannot index in non-increasing order
        self.values_num = self.values_num[index_positions]
        self.values_str = self.values_str[index_positions]

    @abstractmethod
    def insert(self,
               loc: int,
               name: str,
               values: np.ndarray | Iterable | int | float) -> None:
        """
        Insert a column in either the numerical data or the string data, depending on the type of the <values> array.
            The column is inserted at position <loc> with name <name>.
        """

    # endregion

    # region data methods
    def merge(self,
              other: BaseTemporalDataFrame,
              name: str | None = None) -> TemporalDataFrame:
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

        if not np.all(self.columns_num[:] == other.columns_num) and not np.all(
                self.columns_str[:] == other.columns_num):
            raise ValueError("Cannot merge TemporalDataFrames with different columns.")

        if not self.timepoints_column_name == other.timepoints_column_name:
            raise ValueError("Cannot merge TemporalDataFrames with different 'timepoints_column_name'.")

        if self.has_repeating_index is not other.has_repeating_index:
            raise ValueError('Cannot merge TemporalDataFrames if one has repeating index while the other has not.')

        if self.empty:
            combined_index = np.array([])
            for tp in self.time_points:
                combined_index = np.concatenate((combined_index,
                                                 self.index_at(tp),
                                                 other.index_at(tp)))

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

        return dataframe.TemporalDataFrame(data=_data,
                                           repeating_index=self.has_repeating_index,
                                           columns_numerical=self.columns_num[:],
                                           columns_string=self.columns_str[:],
                                           time_list=_time_list,
                                           time_col_name=self.timepoints_column_name,
                                           name=name)

    # endregion


class BaseTemporalDataFrameView(BaseTemporalDataFrame, ABC):
    """
    Abstract base class for views on a TemporalDataFrame.
    """

    # region magic methods
    def __init__(self,
                 parent: BaseTemporalDataFrameImplementation,
                 index_positions: np.ndarray,
                 columns_numerical: np.ndarray,
                 columns_string: np.ndarray,
                 inverted: bool = False):
        self._parent = parent
        self._index_positions = index_positions
        self._columns_numerical = columns_numerical
        self._columns_string = columns_string
        self._repeating_index = parent.has_repeating_index

        self._inverted = inverted

    def __repr__(self) -> str:
        return f"{self.full_name}\n" \
               f"{self.head()}"

    def __dir__(self) -> list[str]:
        return dir(BaseTemporalDataFrameView) + list(map(str, self.columns))

    def __getattr__(self,
                    column_name: str) -> Type[Self]:
        """
        Get a single column from this view of a TemporalDataFrame.
        """
        if column_name in self.columns_num:
            return self.__class__(self._parent, self._index_positions, np.array([column_name]), np.array([]))

        elif column_name in self.columns_str:
            return self.__class__(self._parent, self._index_positions, np.array([]), np.array([column_name]))

        prefix = 'backed ' if self.is_backed else ''
        raise AttributeError(f"'{column_name}' not found in this view of a {prefix}TemporalDataFrame.")

    def __delattr__(self,
                    column_name: str) -> NoReturn:
        raise TypeError('Cannot delete columns from a view.')

    def __getitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER]) -> Type[Self]:
        """
        Get a subset.
        """
        _index_positions, _columns_numerical, _columns_string = self.__parse_inverted(*parse_slicer(self, slicer))

        return self.__class__(self._parent, _index_positions, _columns_numerical, _columns_string,
                              inverted=self._inverted)

    @abstractmethod
    def _setitem_reorder_values(self, _index_positions, index_array, values):
        pass

    @abstractmethod
    def _setitem_set_numerical_values(self, _columns_numerical, _index_positions, columns_array, values):
        pass

    @abstractmethod
    def _setitem_set_string_values(self, _columns_string, _index_positions, columns_array, lcn, values):
        pass

    def __setitem__(self,
                    slicer: SLICER | tuple[SLICER, SLICER] | tuple[SLICER, SLICER, SLICER],
                    values: Number | np.number | str | Collection | BaseTemporalDataFrame) -> None:
        """
        Set values in a subset.
        """
        index_positions, column_num_slicer, column_str_slicer, (tp_array, index_array, columns_array) = \
            parse_slicer(self, slicer)

        _index_positions, _columns_numerical, _columns_string = \
            self.__parse_inverted(index_positions,
                                  column_num_slicer,
                                  column_str_slicer,
                                  (tp_array, index_array, columns_array))

        if self._inverted:
            columns_array = np.concatenate((_columns_numerical, _columns_string))

        elif columns_array is None:
            columns_array = self.columns

        # parse values
        lcn, lcs = len(_columns_numerical), len(_columns_string)

        values = parse_values(values, len(_index_positions), lcn + lcs)

        if not lcn + lcs:
            return

        # reorder values to match original index
        values = self._setitem_reorder_values(_index_positions, index_array, values)

        if lcn:
            self._setitem_set_numerical_values(_columns_numerical, _index_positions, columns_array, values)

        if lcs:
            self._setitem_set_string_values(_columns_string, _index_positions, columns_array, lcn, values)

    def __invert__(self) -> Type[Self]:
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """
        return self.__class__(self._parent, self._index_positions, self._columns_numerical, self._columns_string,
                              inverted=not self._inverted)

    # endregion

    # region attributes
    @property
    def name(self) -> str:
        """
        Get the name.
        """
        return f"view of {self._parent.name}"

    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        parts = []
        if self.empty:
            parts.append('empty')

        if self.is_inverted:
            parts.append('inverted')

        parent_full_name = self._parent.full_name
        if not parent_full_name.startswith('TemporalDataFrame'):
            parent_full_name = parent_full_name[0].lower() + parent_full_name[1:]

        parts += ['view of', parent_full_name]

        parts[0] = parts[0].capitalize()

        return ' '.join(parts)

    @property
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """
        return self._parent.timepoints_column[self.index_positions]

    @property
    def timepoints_column_name(self) -> str | None:
        """
        Get the name of the column containing the time-points values.
        """
        return self._parent.timepoints_column_name

    @property
    def index(self) -> np.ndarray:
        """
        Get the index across all time-points.
        """
        return self._parent.index[self._index_positions].copy()

    @property
    def n_index(self) -> int:
        """
        Get the length of the index.
        """
        return len(self._index_positions)

    @property
    def columns_num(self) -> np.ndarray:
        """
        Get the list of column names for numerical data.
        """
        return self._columns_numerical.copy()

    @property
    def n_columns_num(self) -> int:
        """
        Get the number of numerical data columns.
        """
        return len(self._columns_numerical)

    @property
    def columns_str(self) -> np.ndarray:
        """
        Get the list of column names for string data.
        """
        return self._columns_string.copy()

    @property
    def n_columns_str(self) -> int:
        """
        Get the number of string data columns.
        """
        return len(self._columns_string)

    @property
    def index_positions(self) -> np.ndarray:
        """
        Get the list of indices in this view.
        """
        return self._index_positions

    @property
    def columns_num_positions(self) -> np.ndarray:
        if not self.n_columns_num:
            return np.array([], dtype=int)

        return npi.indices(self._parent.columns_num[:], self.columns_num[:])

    @property
    def columns_str_positions(self) -> np.ndarray:
        if not self.n_columns_str:
            return np.array([], dtype=int)

        return npi.indices(self._parent.columns_str[:], self.columns_str[:])

    @property
    def parent(self) -> BaseTemporalDataFrameImplementation:
        """Get the parent TemporalDataFrame of this view."""
        return self._parent

    # endregion

    # region predicates
    @property
    def has_locked_indices(self) -> bool:
        return self._parent.has_locked_indices

    @property
    def has_locked_columns(self) -> bool:
        return self._parent.has_locked_columns

    @property
    def empty(self) -> bool:
        """
        Is there data ?
        """
        return not self.n_index or not self.n_columns

    @property
    def is_view(self) -> bool:
        """
        Is this a view on a TemporalDataFrame ?
        """
        return True

    @property
    def is_inverted(self) -> bool:
        """
        Whether this view of a TemporalDataFrame is inverted or not.
        """
        return self._inverted

    @property
    def has_repeating_index(self) -> bool:
        return self._repeating_index

    # endregion

    # region methods
    def __parse_inverted(self,
                         index_slicer: np.ndarray,
                         column_num_slicer: np.ndarray,
                         column_str_slicer: np.ndarray,
                         arrays: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp_array, index_array, columns_array = arrays

        if self._inverted:
            if tp_array is None and index_array is None:
                _index_positions = self._index_positions[index_slicer]

            else:
                _index_positions = self._index_positions[~np.isin(self._index_positions, index_slicer)]

            if columns_array is None:
                _columns_numerical = column_num_slicer
                _columns_string = column_str_slicer

            else:
                _columns_numerical = self._columns_numerical[~np.isin(self._columns_numerical, column_num_slicer)]
                _columns_string = self._columns_string[~np.isin(self._columns_string, column_str_slicer)]

            return _index_positions, _columns_numerical, _columns_string

        return self._index_positions[index_slicer], column_num_slicer, column_str_slicer

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
                self.timepoints_column_name is not None else np.concatenate(([DEFAULT_TIME_POINTS_COL_NAME, ''],
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
            # display the first n rows of the first 5 timepoints in this ViewTemporalDataFrame
            repr_string += f"\033[4mTime point : {repr(tp)}\033[0m\n"

            if len(self._columns_numerical) and len(self._columns_string):
                tp_mask = self.get_timepoint_mask(tp)

                tp_numerical_array = self.values_num[tp_mask]
                tp_string_array = self.values_str[tp_mask]

                tp_array = np.array([[tp] for _ in range(min(n, tp_numerical_array.shape[0]))])

                spacer = np.array([['|'] for _ in range(min(n, tp_numerical_array.shape[0]))])

                tp_col_name = DEFAULT_TIME_POINTS_COL_NAME if self.timepoints_column_name is None else \
                    self.timepoints_column_name
                columns = np.concatenate(([tp_col_name, ''], self.columns_num[:], [''], self.columns_str[:]))

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
                nb_rows_at_tp = int(np.sum(self.get_timepoint_mask(tp)))

                tp_array_ = np.array([[tp] for _ in range(min(n, nb_rows_at_tp))])

                spacer_ = np.array([['|'] for _ in range(min(n, nb_rows_at_tp))])

                columns_ = [self.timepoints_column_name, ''] if self.timepoints_column_name is not None \
                    else [DEFAULT_TIME_POINTS_COL_NAME, '']

                tp_df = pd.DataFrame(np.concatenate((tp_array_, spacer_), axis=1),
                                     index=self.index_at(tp)[:n],
                                     columns=columns_)

                tp_shape = tp_df.shape

            # remove unwanted shape display by pandas and replace it by our own
            repr_string += re.sub(r'\\n\[.*$', '', repr(tp_df)) + '\n' + f'[{tp_shape[0]} x {tp_shape[1]}]\n\n'

        # then display only the list of remaining timepoints
        if len(timepoints_list) > 5:
            repr_string += f"\nSkipped time points {repr_array(timepoints_list[5:])} ...\n\n\n"

        return repr_string

    def get_timepoint_mask(self,
                           timepoint: str | TimePoint) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """
        return self._parent.timepoints_column[self.index_positions] == TimePoint(timepoint)

    def index_at(self, timepoint: str | TimePoint) -> np.ndarray:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """
        return self._parent.index[self.index_positions[self.get_timepoint_mask(timepoint)]]

    # endregion

    # region data methods
    def merge(self,
              other: BaseTemporalDataFrame,
              name: str | None = None) -> TemporalDataFrame:
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """
        raise NotImplemented

    # endregion
