# coding: utf-8
# Created on 06/04/2022 11:32
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Union, Optional, Any, Collection
from typing_extensions import Literal

from . import dataframe
from vdata.utils import are_equal
from vdata.time_point import TimePoint, mean as tp_mean
from .name_utils import SLICER, H5Data

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame
    from .view import ViewTemporalDataFrame
    from .indexer import VAtIndexer, ViAtIndexer, VLocIndexer, ViLocIndexer


# ====================================================
# code
class BaseTemporalDataFrame(ABC):

    @abstractmethod
    def __dir__(self) -> 'ViewTemporalDataFrame':
        pass

    @abstractmethod
    def __getattr__(self,
                    column_name: str) -> 'ViewTemporalDataFrame':
        """
        Get a single column.
        """

    @abstractmethod
    def __getitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a subset.
        """

    @abstractmethod
    def __setitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]],
                    values: Union[Number, np.number, str, Collection, 'TemporalDataFrame', 'ViewTemporalDataFrame']) \
            -> None:
        """
        Set values in a subset.
        """

    def _check_compatibility(self,
                             value: 'BaseTemporalDataFrame') -> None:
        # time-points column and nb of columns must be identical
        if not np.array_equal(self.timepoints_column, value.timepoints_column):
            raise ValueError("Time-points do not match.")
        if not np.array_equal(self.n_columns_num, value.n_columns_num):
            raise ValueError("Columns numerical do not match.")
        if not np.array_equal(self.n_columns_str, value.n_columns_str):
            raise ValueError("Columns string do not match.")

    def _add_core(self,
                  value: Union[Number, np.number, str, 'BaseTemporalDataFrame']) -> 'TemporalDataFrame':
        """
        Internal function for adding a value, called from __add__. Do not use directly.
        """
        if isinstance(value, (Number, np.number)):
            if self.values_num.size == 0:
                raise ValueError("No numerical data to add.")

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
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_list=self.timepoints_column,
                                               lock=self.lock,
                                               name=f"{self.full_name} + {value_name}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index, columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_col_name=self.timepoints_column_name,
                                               lock=self.lock,
                                               name=f"{self.full_name} + {value_name}")

    @abstractmethod
    def __add__(self,
                value: Union[Number, np.number, str, 'BaseTemporalDataFrame']) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """

    @abstractmethod
    def __radd__(self,
                 value: Union[Number, np.number, str]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """

    @abstractmethod
    def __iadd__(self,
                 value: Union[Number, np.number, str, 'BaseTemporalDataFrame']) \
            -> Union['TemporalDataFrame', 'ViewTemporalDataFrame']:
        """
        Modify inplace the values :
            - numerical values incremented by <value> if <value> is a number.
            - <value> appended to string values if <value> is a string.
        """

    def _op_core(self,
                 value: Union[Number, np.number, 'BaseTemporalDataFrame'],
                 operation: Literal['sub', 'mul', 'div']) -> 'TemporalDataFrame':
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
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_list=self.timepoints_column,
                                               lock=self.lock,
                                               name=f"{self.full_name} {op} {value_name}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index, columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return dataframe.TemporalDataFrame(df_data,
                                               time_col_name=self.timepoints_column_name,
                                               lock=self.lock,
                                               name=f"{self.full_name} {op} {value_name}")

    @abstractmethod
    def __sub__(self,
                value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """

    @abstractmethod
    def __rsub__(self,
                 value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """

    @abstractmethod
    def __isub__(self,
                 value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> None:
        """
        Modify inplace the values :
            - numerical values decremented by <value>.
        """

    @abstractmethod
    def __mul__(self,
                value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """

    @abstractmethod
    def __rmul__(self,
                 value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """

    @abstractmethod
    def __imul__(self,
                 value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> None:
        """
        Modify inplace the values :
            - numerical values multiplied by <value>.
        """

    @abstractmethod
    def __truediv__(self,
                    value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values divided by <value>.
        """

    @abstractmethod
    def __rtruediv__(self,
                     value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values divided by <value>.
        """

    @abstractmethod
    def __itruediv__(self,
                     value: Union[Number, np.number, 'BaseTemporalDataFrame']) -> None:
        """
        Modify inplace the values :
            - numerical values divided by <value>.
        """

    def _is_equal(self,
                  other: Any) -> Union[bool, np.ndarray]:
        """
        Internal function for testing the equality with another object.
        Do not use directly, it is called by '__eq__()'.
        """
        if isinstance(other, BaseTemporalDataFrame):
            for attr in ['timepoints_column_name', 'has_locked_indices', 'has_locked_columns', 'columns',
                         'timepoints_column', 'index', 'values_num', 'values_str']:
                if not are_equal(getattr(self, attr), getattr(other, attr)):
                    return False

            return True

        if isinstance(other, (Number, np.number)):
            return self.values_num == other

        elif isinstance(other, str):
            return self.values_str == other

        raise ValueError(f"Cannot compare {self.__class__.__name__} object with object of class "
                         f"{other.__class__.__name__}.")

    @abstractmethod
    def __eq__(self,
               other: Any) -> Union[bool, np.ndarray]:
        """
        Test for equality with :
            - another TemporalDataFrame or view of a TemporalDataFrame
            - a single value (either numerical or string)
        """

    @abstractmethod
    def __invert__(self) -> 'ViewTemporalDataFrame':
        """
        Invert the getitem selection behavior : all elements NOT present in the slicers will be selected.
        """

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
    def lock(self) -> tuple[bool, bool]:
        """
        Get the index and columns lock state.
        """

    @property
    @abstractmethod
    def empty(self) -> bool:
        """
        Is data stored ?
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, list[int], int]:
        """
        Get the shape of this TemporalDataFrame as a 3-tuple of :
            - number of time-points
            - number of rows per time-point
            - number of columns
        """

    @abstractmethod
    def head(self,
             n: int = 5) -> str:
        """
        Get a short representation of the first n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the first n rows in this TemporalDataFrame.
        """

    @abstractmethod
    def tail(self,
             n: int = 5) -> str:
        """
        Get a short representation of the last n rows in this TemporalDataFrame.

        Args:
            n: number of rows to print.

        Returns:
            A short string representation of the last n rows in this TemporalDataFrame.
        """

    @property
    @abstractmethod
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this TemporalDataFrame.
        """

    @abstractmethod
    def get_timepoint_mask(self,
                           timepoint: Union[str, TimePoint]) -> np.ndarray:
        """
        Get a boolean mask indicating where in this TemporalDataFrame's the rows' time-point are equal to <timepoint>.

        Args:
            timepoint: A time-point (str or TimePoint object) to get a mask for.

        Returns:
            A boolean mask for rows matching the time-point.
        """

    @property
    @abstractmethod
    def timepoints_column(self) -> np.ndarray:
        """
        Get the column of time-point values.
        """

    @property
    @abstractmethod
    def timepoints_column_str(self) -> np.ndarray:
        """
        Get the column of time-point values cast as strings.
        """

    @property
    @abstractmethod
    def timepoints_column_numerical(self) -> np.ndarray:
        """
        Get the column of time-point values cast as floats.
        """

    @property
    @abstractmethod
    def timepoints_column_name(self) -> Optional[str]:
        """
        Get the name of the column containing the time-points values.
        """

    @property
    @abstractmethod
    def index(self) -> np.ndarray:
        """
        Get the index across all time-points.
        """

    @property
    @abstractmethod
    def n_index(self) -> int:
        """
        Get the length of the index.
        """

    @abstractmethod
    def index_at(self,
                 timepoint: Union[str, TimePoint]) -> np.ndarray:
        """
        Get the index of rows existing at the given time-point.

        Args:
            timepoint: time_point for which to get the index.

        Returns:
            The index of rows existing at that time-point.
        """

    @property
    @abstractmethod
    def columns_num(self) -> np.ndarray:
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
    def columns_str(self) -> np.ndarray:
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
    @abstractmethod
    def values_num(self) -> np.ndarray:
        """
        Get the numerical data.
        """

    @property
    @abstractmethod
    def values_str(self) -> np.ndarray:
        """
        Get the string data.
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

    def _convert_to_pandas(self,
                           with_timepoints: Optional[str] = None,
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
        index_ = self.index
        if str_index:
            index_ = index_.astype(str)

        if with_timepoints is None:
            return pd.concat((pd.DataFrame(self.values_num, index=index_, columns=self.columns_num),
                              pd.DataFrame(self.values_str, index=index_, columns=self.columns_str)),
                             axis=1)

        if timepoints_type == 'string':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_str[:, None], index=index_, columns=[str(with_timepoints)]),
                pd.DataFrame(self.values_num, index=index_, columns=self.columns_num),
                pd.DataFrame(self.values_str, index=index_, columns=self.columns_str)
            ), axis=1)

        elif timepoints_type == 'numerical':
            return pd.concat((
                pd.DataFrame(self.timepoints_column_numerical[:, None], index=index_, columns=[str(with_timepoints)]),
                pd.DataFrame(self.values_num, index=index_, columns=self.columns_num),
                pd.DataFrame(self.values_str, index=index_, columns=self.columns_str)
            ), axis=1)

        raise ValueError(f"Invalid timepoints_type argument '{timepoints_type}'. Should be 'string' or 'numerical'.")

    @abstractmethod
    def to_pandas(self,
                  with_timepoints: Optional[str] = None,
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

    @property
    @abstractmethod
    def at(self) -> 'VAtIndexer':
        """
        Access a single value from a pair of row and column labels.
        """

    @property
    @abstractmethod
    def iat(self) -> 'ViAtIndexer':
        """
        Access a single value from a pair of row and column indices.
        """

    @property
    @abstractmethod
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

    @property
    @abstractmethod
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

    def _min_max_mean_core(self,
                           axis: Optional[int],
                           func: Literal['min', 'max', 'mean']) -> Union[float, 'TemporalDataFrame']:
        if axis is None:
            return getattr(self.values_num, func)()

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
                columns=self.columns_num,
            ),
                time_list=[mmm_tp for _ in enumerate(i0)],
                time_col_name=self.timepoints_column_name)

        elif axis == 1:
            return dataframe.TemporalDataFrame(data=pd.DataFrame(
                getattr(np, func)([self.values_num[self.get_timepoint_mask(tp)] for tp in self.timepoints], axis=1),
                index=[func for _ in enumerate(self.timepoints)],
                columns=self.columns_num
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
            axis: Optional[int] = None) -> Union[float, 'TemporalDataFrame']:
        """
        Get the min value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global min). (default: None)
        """
        return self._min_max_mean_core(axis, 'min')

    def max(self,
            axis: Optional[int] = None) -> Union[float, 'TemporalDataFrame']:
        """
        Get the max value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global max). (default: None)
        """
        return self._min_max_mean_core(axis, 'max')

    def mean(self,
             axis: Optional[int] = None) -> Union[float, 'TemporalDataFrame']:
        """
        Get the mean value along the specified axis.

        Args:
            axis: Can be 0 (time-points), 1 (rows), 2 (columns) or None (global mean). (default: None)
        """
        return self._min_max_mean_core(axis, 'mean')

    @abstractmethod
    def write(self,
              file: Optional[Union[str, Path, H5Data]] = None) -> None:
        """
        Save in HDF5 file format.

        Args:
            file: path to save the data.
        """

    @abstractmethod
    def to_csv(self, path: Union[str, Path], sep: str = ",", na_rep: str = "",
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

    def _copy(self) -> 'TemporalDataFrame':
        """
        Internal function for getting a copy. Do not use directly, it is called by '.copy()'.
        """
        from .dataframe import TemporalDataFrame

        if self.timepoints_column_name is None:
            return TemporalDataFrame(self.to_pandas(),
                                     repeating_index=self._repeating_index,
                                     time_list=self.timepoints_column,
                                     lock=self.lock,
                                     name=f"copy of {self.name}")

        return TemporalDataFrame(self.to_pandas(with_timepoints=self.timepoints_column_name),
                                 repeating_index=self._repeating_index,
                                 time_col_name=self.timepoints_column_name,
                                 lock=self.lock,
                                 name=f"copy of {self.name}")

    @abstractmethod
    def copy(self) -> 'TemporalDataFrame':
        """
        Get a copy.
        """

    @abstractmethod
    def merge(self,
              other: Union['TemporalDataFrame', 'ViewTemporalDataFrame'],
              name: Optional[str] = None) -> 'TemporalDataFrame':
        """
        Merge two TemporalDataFrames together, by rows. The column names and time points must match.

        Args:
            other: a TemporalDataFrame to merge with this one.
            name: a name for the merged TemporalDataFrame.

        Returns:
            A new merged TemporalDataFrame.
        """
        # TODO : test for implementations
