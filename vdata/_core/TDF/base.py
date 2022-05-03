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

from typing import TYPE_CHECKING, Union, Optional, Any
from typing_extensions import Literal

from vdata.time_point import TimePoint
from .name_utils import SLICER, H5Data
from .utils import are_equal

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
                    values: np.ndarray) -> None:
        """
        Set values in a subset.
        """

    def _add_core(self,
                  value: Union[Number, np.number, str]) -> 'TemporalDataFrame':
        """
        Internal function for adding a value, called from __add__. Do not use directly.
        """
        from .dataframe import TemporalDataFrame

        if isinstance(value, (Number, np.number)):
            if self.values_num.size == 0:
                raise ValueError("No numerical data to add.")

            values_num = self.values_num + value
            values_str = self.values_str

        else:
            if self.values_str.size == 0:
                raise ValueError("No string data to add to.")

            values_num = self.values_num
            values_str = np.char.add(self.values_str, value)

        if self.timepoints_column_name is None:
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return TemporalDataFrame(df_data,
                                     time_list=self.timepoints_column,
                                     lock=self.lock,
                                     name=f"{self.name} + {value}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index, columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return TemporalDataFrame(df_data,
                                     time_col_name=self.timepoints_column_name,
                                     lock=self.lock,
                                     name=f"{self.name} + {value}")

    @abstractmethod
    def __add__(self,
                value: Union[Number, np.number, str]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values incremented by <value> if <value> is a number
            - <value> appended to string values if <value> is a string
        """

    @abstractmethod
    def __iadd__(self,
                 value: Union[Number, np.number, str]) -> Union['TemporalDataFrame', 'ViewTemporalDataFrame']:
        """
        Modify inplace the values :
            - numerical values incremented by <value> if <value> is a number.
            - <value> appended to string values if <value> is a string.
        """

    def _op_core(self,
                 value: Union[Number, np.number],
                 operation: Literal['sub', 'mul', 'div']) -> 'TemporalDataFrame':
        """
        Internal function for subtracting, multiplying by and dividing by a value, called from __add__. Do not use
        directly.
        """
        from .dataframe import TemporalDataFrame

        if operation == 'sub':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to subtract.")
            op = '-'
            values_num = self.values_num - value

        elif operation == 'mul':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to multiply.")
            op = '*'
            values_num = self.values_num * value

        elif operation == 'div':
            if self.values_num.size == 0:
                raise ValueError("No numerical data to divide.")
            op = '/'
            values_num = self.values_num / value

        else:
            raise ValueError(f"Unknown operation '{operation}'.")

        if self.timepoints_column_name is None:
            df_data = pd.concat((pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return TemporalDataFrame(df_data,
                                     time_list=self.timepoints_column,
                                     lock=self.lock,
                                     name=f"{self.name} {op} {value}")

        else:
            df_data = pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                              index=self.index, columns=[str(self.timepoints_column_name)]),
                                 pd.DataFrame(values_num, index=self.index, columns=self.columns_num),
                                 pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                                axis=1)

            return TemporalDataFrame(df_data,
                                     time_col_name=self.timepoints_column_name,
                                     lock=self.lock,
                                     name=f"{self.name} {op} {value}")

    @abstractmethod
    def __sub__(self,
                value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values decremented by <value>.
        """

    @abstractmethod
    def __isub__(self,
                 value: Union[Number, np.number]) -> None:
        """
        Modify inplace the values :
            - numerical values decremented by <value>.
        """

    @abstractmethod
    def __mul__(self,
                value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values multiplied by <value>.
        """

    @abstractmethod
    def __imul__(self,
                 value: Union[Number, np.number]) -> None:
        """
        Modify inplace the values :
            - numerical values multiplied by <value>.
        """

    @abstractmethod
    def __truediv__(self,
                    value: Union[Number, np.number]) -> 'TemporalDataFrame':
        """
        Get a copy with :
            - numerical values divided by <value>.
        """

    @abstractmethod
    def __itruediv__(self,
                     value: Union[Number, np.number]) -> None:
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

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name.
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

    def _convert_to_pandas(self,
                           with_timepoints: Optional[str] = None,
                           str_index: bool = False) -> pd.DataFrame:
        """
        Internal function for converting to a pandas DataFrame. Do not use directly, it is called by '.to_pandas()'.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
            str_index: cast index as string ?
        """
        index_ = self.index
        if str_index:
            index_ = index_.astype(str)

        if with_timepoints is None:
            return pd.concat((pd.DataFrame(self.values_num, index=index_, columns=self.columns_num),
                              pd.DataFrame(self.values_str, index=index_, columns=self.columns_str)),
                             axis=1)

        return pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                       index=index_, columns=[str(with_timepoints)]),
                          pd.DataFrame(self.values_num, index=index_, columns=self.columns_num),
                          pd.DataFrame(self.values_str, index=index_, columns=self.columns_str)),
                         axis=1)

    @abstractmethod
    def to_pandas(self,
                  with_timepoints: Optional[str] = None,
                  str_index: bool = False) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
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
