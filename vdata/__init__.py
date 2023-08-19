"""Annotated, temporal and multivariate observation data."""

from ch5mpy import H5Mode

from vdata.data import VData, VDataView, concatenate, convert_anndata_to_vdata
from vdata.IO import (
    IncoherenceError,
    ShapeError,
    VLockError,
    getLoggingLevel,
    setLoggingLevel,
)
from vdata.tdf import Index, TemporalDataFrame, TemporalDataFrameView
from vdata.timepoint import TimePoint

read = VData.read
read_from_csv = VData.read_from_csv
read_from_anndata = VData.read_from_anndata

mode = H5Mode

__all__ = [
    "VData",
    "TemporalDataFrame",
    "VDataView",
    "TemporalDataFrameView",
    "convert_anndata_to_vdata",
    "setLoggingLevel",
    "getLoggingLevel",
    "concatenate",
    "ShapeError",
    "IncoherenceError",
    "VLockError",
    "TimePoint",
    "Index",
]
