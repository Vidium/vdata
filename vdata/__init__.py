"""Annotated, temporal and multivariate observation data."""

from ch5mpy import H5Mode

from vdata.data import VData, VDataView, concatenate, convert_anndata_to_vdata
from vdata.IO import (
    IncoherenceError,
    ShapeError,
    VBaseError,
    VLockError,
)
from vdata.tdf import RepeatingIndex, TemporalDataFrame, TemporalDataFrameView
from vdata.timepoint import TimePoint
from vdata.update import CURRENT_VERSION, update_vdata

read = VData.read
read_from_csv = VData.read_from_csv
read_from_anndata = VData.read_from_anndata
read_from_pickle = VData.read_from_pickle

mode = H5Mode

__all__ = [
    "VData",
    "TemporalDataFrame",
    "VDataView",
    "TemporalDataFrameView",
    "convert_anndata_to_vdata",
    "concatenate",
    "VBaseError",
    "ShapeError",
    "IncoherenceError",
    "VLockError",
    "TimePoint",
    "RepeatingIndex",
    "CURRENT_VERSION",
    "update_vdata",
]
