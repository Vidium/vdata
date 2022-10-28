# coding: utf-8
# Created on 31/03/2022 15:20
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy_indexed as npi

from .base import BaseTemporalDataFrameView


# ====================================================
# code
class TemporalDataFrameView(BaseTemporalDataFrameView):
    """A view on a regular TemporalDataFrame."""

    # region magic methods
    def _setitem_reorder_values(self, _index_positions, index_array, values):
        if index_array is not None:
            _index_positions.sort()

            original_positions = self._parent._get_index_positions(index_array)
            values = values[np.argsort(npi.indices(_index_positions,
                                                   original_positions[np.isin(original_positions, _index_positions)]))]
        return values

    def _setitem_set_numerical_values(self, _columns_numerical, _index_positions, columns_array, values):
        self._parent.values_num[_index_positions[:, None],
                                npi.indices(self._parent.columns_num[:], _columns_numerical)] = \
            values[:, npi.indices(columns_array, _columns_numerical)].astype(float)

    def _setitem_set_string_values(self, _columns_string, _index_positions, columns_array, lcn, values):
        # cast values as string
        values_str = values[:, npi.indices(columns_array, _columns_string)].astype(str)

        # cast string array to larger str dtype if needed
        if values_str.dtype > self._parent.values_str.dtype:
            self._parent.values_str = self._parent.values_str.astype(values_str.dtype)

        # assign values into array
        self._parent.values_str[_index_positions[:, None],
                                npi.indices(self._parent.columns_str[:], _columns_string)] = values_str

    # endregion

    # region attributes
    @property
    def timepoints(self) -> np.ndarray:
        """
        Get the list of unique time points in this ViewTemporalDataFrame.
        """
        return np.unique(self.timepoints_column)

    @property
    def values_num(self) -> np.ndarray:
        """
        Get the numerical data.
        """
        return self._parent.values_num[self.index_positions[:, None], self.columns_num_positions]

    @values_num.setter
    def values_num(self,
                   values: np.ndarray) -> None:
        """
        Set the numerical data.
        """
        self._parent.values_num[self.index_positions[:, None], self.columns_num_positions] = values

    @property
    def values_str(self) -> np.ndarray:
        """
        Get the string data.
        """
        return self._parent.values_str[self.index_positions[:, None], self.columns_str_positions]

    @values_str.setter
    def values_str(self,
                   values: np.ndarray) -> None:
        """
        Set the string data.
        """
        self._parent.values_str = self._parent.values_str.astype(values.dtype)
        self._parent.values_str[self.index_positions[:, None], self.columns_str_positions] = values

    # endregion
