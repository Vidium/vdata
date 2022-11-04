# coding: utf-8
# Created on 22/10/2022 15:35
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from vdata.core.dataset_proxy.base import _Dataset1DMixin, _NumDatasetProxy, _StrDatasetProxy, _TPDatasetProxy


# ====================================================
# code
class NumDatasetProxy1D(_Dataset1DMixin, _NumDatasetProxy):
    """Simple proxy for 1D numerical h5py.Dataset objects for performing inplace operations."""


class StrDatasetProxy1D(_Dataset1DMixin, _StrDatasetProxy):
    """Simple proxy for 1D string h5py.Dataset objects for performing inplace operations."""

    @property
    def dtype(self) -> np.dtype:
        longest = len(max(self._data, key=len)) if self._data.size else 0
        return np.dtype(f'<U{longest}')


class TPDatasetProxy1D(_Dataset1DMixin, _TPDatasetProxy):
    """Simple proxy for 1D TimePoint objects for performing inplace operations."""
