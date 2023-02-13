# coding: utf-8
# Created on 22/10/2022 20:01
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from vdata.core.dataset_proxy import DatasetProxy
from vdata.core.tdf.backed_tdf.meta import CheckH5File
from vdata.core.tdf.base import BaseTemporalDataFrame
from vdata.h5pickle.name_utils import H5Mode


# ====================================================
# code
class BackedMixin(BaseTemporalDataFrame, metaclass=CheckH5File):
    """
    Abstract base mixin class for TemporalDataFrames backed on a h5 file.
    /!\ Inherit from this Mixin FIRST to properly override the `is_backed` predicate.
    """

    # region magic methods
    def _iadd_str(self,
                  value: str) -> BackedMixin:
        """Inplace modification of the string values."""
        self.dataset_str += value
        return self

    # endregion

    # region attributes
    @property
    @abstractmethod
    def h5_mode(self) -> H5Mode:
        """Get the mode the h5 file was opened with."""

    @property
    def dataset_num(self) -> DatasetProxy:
        """Get the numerical data as a dataset proxy for efficient computations."""
        return self._numerical_array

    @dataset_num.setter
    def dataset_num(self, value: np.ndarray | DatasetProxy):
        self.values_num = value

    @property
    def dataset_str(self) -> DatasetProxy:
        """Get the string data as a dataset proxy for efficient computations."""
        return self._string_array

    @dataset_str.setter
    def dataset_str(self, value: np.ndarray | DatasetProxy):
        self.values_str = value

    # endregion

    # region predicates
    @property
    def is_backed(self) -> bool:
        """
        Is this TemporalDataFrame backed on a file ?
        """
        return True

    @property
    @abstractmethod
    def is_closed(self) -> bool:
        """
        Is the h5 file (this TemporalDataFrame is backed on) closed ?
        """

    # endregion
