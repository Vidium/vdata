# coding: utf-8
# Created on 22/10/2022 15:37
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from vdata.core.dataset_proxy.base import _Dataset2DMixin, _NumDatasetProxy, _StrDatasetProxy


# ====================================================
# code
class NumDatasetProxy2D(_Dataset2DMixin, _NumDatasetProxy):
    """Simple proxy for 2D numerical h5py.Dataset objects for performing inplace operations."""


class StrDatasetProxy2D(_Dataset2DMixin, _StrDatasetProxy):
    """Simple proxy 2D for string h5py.Dataset objects for performing inplace operations."""
