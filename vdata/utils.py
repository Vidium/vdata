from __future__ import annotations

import builtins
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Collection, Mapping, TypeGuard, TypeVar

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp
from vdata.array_view import NDArrayView
from vdata.IO.errors import ShapeError

if TYPE_CHECKING:
    from vdata._typing import AnyNDArrayLike_IFS, NDArray_IFS, NDArrayLike_IFS, PreSlicer
    

_builtin_names = dir(builtins)
_builtin_names.remove('False')
_builtin_names.remove('True')
_builtin_names.remove('None')

_V = TypeVar('_V')

# region misc -----------------------------------------------------------------
def first_in(d: Mapping[Any, _V]) -> _V:
    return next(iter(d.values()))


def isCollection(obj: Any) -> TypeGuard[Collection[Any]]:
    """
    Whether an object is a collection.
    :param obj: an object to test.
    :return: whether an object is a collection.
    """
    return isinstance(obj, Collection) and not isinstance(obj, (str, bytes, bytearray, memoryview))


def are_equal(obj1: Any,
              obj2: Any) -> bool:    
    if isinstance(obj1, (np.ndarray, ch.H5Array, NDArrayView)):
        if isinstance(obj2, (np.ndarray, ch.H5Array, NDArrayView)):
            return np.array_equal(obj1[:], obj2[:])

        return False

    return bool(obj1 == obj2)


def spacer(nb: int) -> str:
    return "  "*(nb-1) + "  " + u'\u21B3' + " " if nb else ''


def obj_as_str(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return arr.astype(str) if arr.dtype == object else arr

# endregion


# region Representation --------------------------------------------------------------
def repr_array(arr: Any) -> str:
    """Get a short string representation of an array."""
    if isinstance(arr, slice) or arr is Ellipsis or not isCollection(arr):
        return str(arr)

    if isinstance(arr, range) or len(arr) <= 4:
        return f"{str(arr)} ({len(arr)} value{'' if len(arr) == 1 else 's'} long)"

    arr = list(arr)
    return f"[{arr[0]} {arr[1]} ... {arr[-2]} {arr[-1]}] ({len(arr)} values long)"

# endregion


# region Type coercion ---------------------------------------------------------------
def deep_dict_convert(obj: Mapping[Any, Any]) -> dict[Any, Any]:
    """
    'Deep' convert a mapping of any kind (and children mappings) into regular dictionaries.

    Args:
        obj: a mapping to convert.

    Returns:
        a converted dictionary.
    """
    if not isinstance(obj, Mapping):
        return obj

    return {k: deep_dict_convert(v) for k, v in obj.items()}

# endregion


# Formatting & Conversion ------------------------------------------------
def slice_or_range_to_list(s: slice | range, _c: Collection[Any]) -> list[Any]:
    """
    Converts a slice or a range to a list of elements within that slice.
    :param s: a slice or range to convert.
    :param _c: a collection of elements to slice.
    """
    c = np.array(_c)
    if c.ndim != 1:
        raise ShapeError(f"The collection is {c.ndim}D, should be a 1D array.")

    sliced_list = []
    found_start = False
    current_step = 1

    start = s.start
    end = s.stop

    # get step value
    if s.step is None:
        step = 1

    else:
        step = s.step
        if not isinstance(step, int):
            raise ValueError(f"The 'step' value is {step}, should be an int.")

        if step == 0:
            raise ValueError("The 'step' value cannot be 0.")

    if step < 0:
        c = np.flip(c)

    # scan the collection of elements to extract values in the slice/range
    for element in c:
        if not found_start:
            # scan collection until the start element is found
            if element == start:
                sliced_list.append(element)
                found_start = True

        else:
            if element == end:
                break

            elif current_step == step:
                current_step = 1
                sliced_list.append(element)

            else:
                current_step += 1

    return sliced_list


def slicer_to_array(slicer: PreSlicer,
                    reference_index: AnyNDArrayLike_IFS | tp.TimePointArray) -> NDArray_IFS | None:
    """
    Format a slicer into an array of allowed values given in the 'reference_index' parameter.

    Args:
        slicer: a PreSlicer object to format.
        reference_index: a collection of allowed values for the slicer.
        on_time_point: slicing on time points ?

    Returns:
        An array of allowed values in the slicer.
    """
    if slicer == slice(None, None, None) or isinstance(slicer, EllipsisType):
        return None

    if isinstance(slicer, (slice, range)):
        return np.array(slice_or_range_to_list(slicer, reference_index))
    
    if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
        return np.array(reference_index)[np.where(slicer.flatten())]

    if not isCollection(slicer):
        return np.array([slicer]) if slicer in reference_index else np.array([])
        
    return np.array(slicer)[np.where(np.in1d(slicer, reference_index))]     # type: ignore[arg-type]


def reformat_index(index: PreSlicer |
                          tuple[PreSlicer] |
                          tuple[PreSlicer, PreSlicer] |
                          tuple[PreSlicer, PreSlicer, PreSlicer],
                   timepoints_reference: tp.TimePointArray,
                   obs_reference: AnyNDArrayLike_IFS,
                   var_reference: NDArrayLike_IFS) \
        -> tuple[tp.TimePointArray | None, 
                 NDArray_IFS | None,
                 NDArray_IFS | None]:
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
        
    index = index + (...,) * (3 - len(index))
    _tp_slicer = slicer_to_array(index[0], timepoints_reference)
        
    return None if _tp_slicer is None else tp.as_timepointarray(_tp_slicer), \
        slicer_to_array(index[1], obs_reference), \
        slicer_to_array(index[2], var_reference)
    

# Representation ---------------------------------------------------------
def repr_index(index: None |
                      PreSlicer |
                      tuple[PreSlicer | None] |
                      tuple[PreSlicer | None, PreSlicer | None] |
                      tuple[PreSlicer | None, PreSlicer | None, PreSlicer | None]) \
        -> str:
    """Get a short string representation of a sub-setting index."""
    if isinstance(index, tuple):
        repr_string = f"Index of {len(index)} element{'' if len(index) == 1 else 's'} : "

        for element in index:
            repr_string += f"\n  \u2022 {repr_array(element) if isCollection(element) else element}"

        return repr_string

    return f"Index of 1 element : \n" \
            f"  \u2022 {repr_array(index) if isCollection(index) else index}"

