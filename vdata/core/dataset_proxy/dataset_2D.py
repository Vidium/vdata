# coding: utf-8
# Created on 22/10/2022 15:37
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from vdata.core.dataset_proxy.base import _Dataset2DMixin, _NumDatasetProxy, _StrDatasetProxy


# ====================================================
# code
class NumDatasetProxy2D(_Dataset2DMixin, _NumDatasetProxy):
    """Simple proxy for 2D numerical h5py.Dataset objects for performing inplace operations."""


class StrDatasetProxy2D(_Dataset2DMixin, _StrDatasetProxy):
    """Simple proxy 2D for string h5py.Dataset objects for performing inplace operations."""

    @property
    def dtype(self) -> np.dtype:
        longest = len(max(self._data[:].flat, key=len)) if self._data.size else 0
        return np.dtype(f'<U{longest}')
