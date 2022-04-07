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

from typing import TYPE_CHECKING, Union, Optional
from typing_extensions import Literal

from vdata.new_time_point import TimePoint
from .name_utils import SLICER, H5Data

if TYPE_CHECKING:
    from .dataframe import TemporalDataFrame
    from .view import ViewTemporalDataFrame


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

    # TODO for views
    # @abstractmethod
    # def __setattr__(self,
    #                 name: str,
    #                 values: np.ndarray) -> None:
    #     """
    #     Set values of a single column. If the column does not already exist, it is appended at the end.
    #     """
    #
    # @abstractmethod
    # def __delattr__(self,
    #                 column_name: str) -> None:
    #     """
    #     Delete a single column.
    #     """

    @abstractmethod
    def __getitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a subset.
        """

    # TODO
    # @abstractmethod
    # def __setitem__(self,
    #                 slicer: Union[SLICER,
    #                               tuple[SLICER, SLICER],
    #                               tuple[SLICER, SLICER, SLICER]],
    #                 values: np.ndarray) -> None:
    #     """
    #     Set values in a subset.
    #     """

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
            values_str = np.core.defchararray.add(self.values_str, value)

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
                 value: Union[Number, np.number, str]) -> None:
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
    def __idiv__(self,
                 value: Union[Number, np.number]) -> None:
        """
        Modify inplace the values :
            - numerical values divided by <value>.
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
                           with_timepoints: Optional[str] = None) -> pd.DataFrame:
        """
        Internal function for converting to a pandas DataFrame. Do not use directly, it is called by '.to_pandas()'.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
        """
        if with_timepoints is None:
            return pd.concat((pd.DataFrame(self.values_num, index=self.index, columns=self.columns_num),
                              pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                             axis=1)

        return pd.concat((pd.DataFrame(self.timepoints_column_str[:, None],
                                       index=self.index, columns=[str(with_timepoints)]),
                          pd.DataFrame(self.values_num, index=self.index, columns=self.columns_num),
                          pd.DataFrame(self.values_str, index=self.index, columns=self.columns_str)),
                         axis=1)

    @abstractmethod
    def to_pandas(self,
                  with_timepoints: Optional[str] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Args:
            with_timepoints: Name of the column containing time-points data to add to the DataFrame. If left to None,
                no column is created.
        """

    @abstractmethod
    def write(self,
              file: Optional[Union[str, Path, H5Data]] = None) -> None:
        """
        Save in HDF5 file format.

        Args:
            file: path to save the data.
        """

    def _copy(self) -> 'TemporalDataFrame':
        """
        Internal function for getting a copy. Do not use directly, it is called by '.copy()'.
        """
        from .dataframe import TemporalDataFrame

        if self.timepoints_column_name is None:
            return TemporalDataFrame(self.to_pandas(),
                                     time_list=self.timepoints_column,
                                     lock=self.lock,
                                     name=f"copy of {self.name}")

        return TemporalDataFrame(self.to_pandas(with_timepoints=self.timepoints_column_name),
                                 time_col_name=self.timepoints_column_name,
                                 lock=self.lock,
                                 name=f"copy of {self.name}")

    @abstractmethod
    def copy(self) -> 'TemporalDataFrame':
        """
        Get a copy.
        """