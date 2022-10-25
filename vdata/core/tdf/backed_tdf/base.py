# coding: utf-8
# Created on 22/10/2022 20:01
# Author : matteo
from __future__ import annotations

from abc import abstractmethod

from typing_extensions import Self

from vdata.core.tdf.backed_tdf.meta import CheckH5File
from vdata.core.tdf.base import BaseTemporalDataFrame


# ====================================================
# imports

# ====================================================
# code
class BackedMixin(BaseTemporalDataFrame, metaclass=CheckH5File):
    """
    Abstract base mixin class for TemporalDataFrames backed on a h5 file.
    /!\ Inherit from this Mixin FIRST to properly override the `is_backed` predicate.
    """

    # region magic methods
    def _iadd_str(self,
                  value: str) -> Self:
        """Inplace modification of the string values."""
        self.values_str += value
        return self

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
