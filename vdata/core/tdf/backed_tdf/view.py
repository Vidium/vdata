# coding: utf-8
# Created on 17/10/2022 19:13
# Author : matteo


# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy_indexed as npi

from vdata.name_utils import H5Mode
from vdata.time_point import TimePoint
from vdata.core.dataset_proxy import DatasetProxy, issubdtype, num_, str_
from vdata.core.tdf.base import BaseTemporalDataFrameView, BaseTemporalDataFrameImplementation
from vdata.core.tdf.backed_tdf.base import BackedMixin

# ====================================================
# code
_CHECK_READ = ('__getattr__', '__getitem__', '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
               '__truediv__', '__rtruediv__', '__invert__', 'name')
_CHECK_WRITE = ('__setitem__', '__iadd__', '__isub__', '__imul__', '__itruediv__')


class BackedTemporalDataFrameView(BackedMixin, BaseTemporalDataFrameView,
                                  read=_CHECK_READ, write=_CHECK_WRITE):
    """A view on a backed TemporalDataFrame."""

    # region magic methods
    def __init__(self,
                 parent: BaseTemporalDataFrameImplementation,
                 index_positions: np.ndarray,
                 columns_numerical: np.ndarray,
                 columns_string: np.ndarray,
                 inverted: bool = False):
        super().__init__(parent, index_positions, columns_numerical, columns_string, inverted)

        self._numerical_array = DatasetProxy(parent.values_num,
                                             view_on=(self.index_positions, self.columns_num_positions))
        self._string_array = DatasetProxy(parent.values_str,
                                          view_on=(self.index_positions, self.columns_str_positions))

    def __repr__(self) -> str:
        if self.is_closed:
            return self.full_name

        return super().__repr__()

    def _setitem_reorder_values(self, _index_positions, index_array, values):
        if index_array is None:
            index_array = self.index_at(self.timepoints[0]) if self.has_repeating_index else self.index

        _index_positions.sort()

        original_positions = self._parent._get_index_positions(index_array)
        values = values[np.argsort(npi.indices(_index_positions,
                                               original_positions[np.isin(original_positions, _index_positions)]))]
        return values

    # endregion

    # region attributes
    @property
    def full_name(self) -> str:
        """
        Get the full name.
        """
        if self.is_closed:
            return "View of TemporalDataFrame backed on closed file."

        return super().full_name

    @property
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this ViewTemporalDataFrame.
        """
        return np.unique(self.timepoints_column)

    @property
    def values_num(self) -> DatasetProxy:
        """
        Get the numerical data.
        """
        return self._numerical_array

    @values_num.setter
    def values_num(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the numerical data.
        """
        if isinstance(values, DatasetProxy) and issubdtype(DatasetProxy.dtype, num_):
            self._numerical_array = values
            return

        self._numerical_array[:] = values

    @property
    def values_str(self) -> DatasetProxy:
        """
        Get the string data.
        """

        return self._string_array

    @values_str.setter
    def values_str(self,
                   values: np.ndarray | DatasetProxy) -> None:
        """
        Set the string data.
        """
        if isinstance(values, DatasetProxy) and DatasetProxy.dtype == str_:
            self._string_array = values
            return

        self._string_array[:] = values

    @property
    def h5_mode(self) -> H5Mode:
        """Get the mode the h5 file was opened with."""
        return self._parent.mode

    # endregion

    # region predicates
    @property
    def is_closed(self) -> bool:
        """
        Is the h5 file (this TemporalDataFrame is backed on) closed ?
        """
        return self._parent.is_closed

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
        return self._parent.timepoints_column[self.index_positions] == str(TimePoint(timepoint))

    # endregion
