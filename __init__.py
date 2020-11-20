# coding: utf-8
# Created on 11/4/20 10:35 AM
# Author : matteo

"""Annotated multivariate observation data with time dimension."""

# ====================================================
# imports
from ._core.vdata import VData
from ._IO.read import read, read_from_GPU

__all__ = ["VData", "read", "read_from_GPU"]
