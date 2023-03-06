# coding: utf-8
# Created on 17/10/2022 19:13
# Author : matteo


# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy_indexed as npi
from ch5mpy import H5Mode
from ch5mpy import H5Array
from numbers import Number

from vdata.time_point import TimePoint
from vdata.core.tdf.base import BaseTemporalDataFrameView
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

    def _setitem_set_numerical_values(self, _columns_numerical, _index_positions, columns_array, values):
        self._parent.dataset_num[np.ix_(_index_positions,
                                        npi.indices(self._parent.columns_num, _columns_numerical))] = \
            values[:, npi.indices(columns_array, _columns_numerical)].astype(float)

    def _setitem_set_string_values(self, _columns_string, _index_positions, columns_array, lcn, values):
        self._parent.dataset_str[np.ix_(_index_positions,
                                        npi.indices(self._parent.columns_str, _columns_string))] = \
            values[:, npi.indices(columns_array, _columns_string)].astype(str)

        # # cast values as string
        # values_str = values[:, npi.indices(columns_array, _columns_string)].astype(str)
        #
        # # cast string array to larger str dtype if needed
        # if values_str.dtype > self._parent.values_str.dtype:
        #     self._parent.values_str = self._parent.values_str.astype(values_str.dtype)
        #
        # # assign values into array
        # self._parent.dataset_str[_index_positions[:, None],
        #                          npi.indices(self._parent.columns_str[:], _columns_string)] = values_str

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
    def values_num(self) -> H5Array[Number]:
        """
        Get the numerical data.
        """
        return self._numerical_array

    @values_num.setter
    def values_num(self,
                   values: np.ndarray | H5Array[Number]) -> None:
        """
        Set the numerical data.
        """
        if isinstance(values, H5Array) and np.issubdtype(values.dtype, np.number):
            self._numerical_array = values

        else:
            self._numerical_array[:] = values

    @property
    def values_str(self) -> H5Array[str]:
        """
        Get the string data.
        """

        return self._string_array

    @values_str.setter
    def values_str(self,
                   values: np.ndarray | H5Array[str]) -> None:
        """
        Set the string data.
        """
        if isinstance(values, H5Array) and np.issubdtype(values.dtype, np.str_):
            self._string_array = values

        else:
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
