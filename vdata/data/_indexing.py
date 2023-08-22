from types import EllipsisType
from typing import Any, Collection

import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp
from vdata._typing import AnyNDArrayLike_IFS, NDArray_IFS, NDArrayLike_IFS, PreSlicer
from vdata.array_view import NDArrayView
from vdata.IO.errors import ShapeError
from vdata.tdf import Index
from vdata.utils import isCollection


def slice_or_range_to_list(s: slice | range, _c: Collection[Any]) -> npt.NDArray[Any]:
    """
    Converts a slice or a range to a list of elements within that slice.

    Args:
        s: a slice or range to convert.
        _c: a collection of elements to slice.
    """
    c = np.asarray(_c)
    if c.ndim != 1:
        raise ShapeError(f"The collection is {c.ndim}D, should be a 1D array.")

    return c[s]


def slicer_to_array(slicer: PreSlicer, reference_index: AnyNDArrayLike_IFS | tp.TimePointArray) -> NDArray_IFS | None:
    """
    Format a slicer into an array of allowed values given in the 'reference_index' parameter.

    Args:
        slicer: a PreSlicer object to format.
        reference_index: a collection of allowed values for the slicer.
        on_time_point: slicing on time points ?

    Returns:
        An array of allowed values in the slicer.
    """
    _is_whole_slice = isinstance(slicer, slice) and slicer == slice(None, None, None)
    if _is_whole_slice or isinstance(slicer, EllipsisType):
        return None

    if isinstance(slicer, (slice, range)):
        return slice_or_range_to_list(slicer, reference_index)

    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return np.array(reference_index)[np.where(slicer.flatten())]

    if isinstance(slicer, tp.TimePointArray):
        return slicer[np.where(np.in1d(slicer, reference_index))]

    if not isCollection(slicer):
        return np.array([slicer]) if slicer in reference_index else np.array([])

    slicer = np.array(slicer)

    if slicer.dtype == bool:
        return reference_index[slicer]

    return slicer[np.where(np.in1d(slicer, reference_index))]


def _gets_whole_axis(slicer: PreSlicer) -> bool:
    return (isinstance(slicer, slice) and slicer == slice(None)) or (
        isinstance(slicer, np.ndarray) and slicer.dtype == bool and bool(np.all(slicer))
    )


def reformat_index(
    index: PreSlicer | tuple[PreSlicer] | tuple[PreSlicer, PreSlicer] | tuple[PreSlicer, PreSlicer, PreSlicer],
    timepoints_reference: tp.TimePointArray | NDArrayView[tp.TimePoint],
    obs_reference: Index | NDArray_IFS,
    var_reference: NDArrayLike_IFS,
) -> tuple[tp.TimePointArray | None, NDArray_IFS | None, NDArray_IFS | None] | None:
    """
    Format a sub-setting index into 3 arrays of selected (and allowed) values for time points, observations and
    variables. The reference collections are used to transform a PreSlicer into an array of selected values.

    Args:
        index: an index to format.
        timepoints_reference: a collection of allowed values for the time points.
        obs_reference: a collection of allowed values for the observations.
        var_reference: a collection of allowed values for the variables.

    Returns:
        3 arrays of selected (and allowed) values for time points, observations and variables.
    """
    if not isinstance(index, tuple):
        index = (index,)

    if all(_gets_whole_axis(i) for i in index):
        return None

    index = index + (...,) * (3 - len(index))
    _tp_slicer = slicer_to_array(index[0], timepoints_reference)

    return (
        None if _tp_slicer is None else tp.as_timepointarray(_tp_slicer),
        slicer_to_array(index[1], obs_reference),
        slicer_to_array(index[2], var_reference),
    )
