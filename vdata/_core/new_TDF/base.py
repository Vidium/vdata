# coding: utf-8
# Created on 06/04/2022 11:32
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING, Union, Optional

from vdata.new_time_point import TimePoint
from .name_utils import SLICER, H5Data

if TYPE_CHECKING:
    from .view import ViewTemporalDataFrame


# ====================================================
# code
class BaseTemporalDataFrame(ABC):

    @abstractmethod
    def __dir__(self) -> 'ViewTemporalDataFrame':
        pass

    # TODO for views
    # def __getattr__(self,
    #                 column_name: str) -> 'ViewTemporalDataFrame':
    #     """
    #     Get a single column.
    #     """
    #
    # def __setattr__(self,
    #                 name: str,
    #                 values: np.ndarray) -> None:
    #     """
    #     Set values of a single column. If the column does not already exist, it is appended at the end.
    #     """
    #
    # def __delattr__(self,
    #                 column_name: str) -> None:
    #     """
    #     Delete a single column.
    #     """

    def __getitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]]) \
            -> 'ViewTemporalDataFrame':
        """
        Get a subset.
        """

    def __setitem__(self,
                    slicer: Union[SLICER,
                                  tuple[SLICER, SLICER],
                                  tuple[SLICER, SLICER, SLICER]],
                    values: np.ndarray) -> None:
        """
        Set values in a subset.
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