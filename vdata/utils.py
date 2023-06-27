from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Mapping, TypeGuard, TypeVar

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

from vdata.array_view import NDArrayView

if TYPE_CHECKING:
    from vdata._typing import PreSlicer

_V = TypeVar('_V')

# region misc -----------------------------------------------------------------
def first_in(d: Mapping[Any, _V]) -> _V:
    return next(iter(d.values()))


def isCollection(obj: Any) -> TypeGuard[Collection[Any]]:
    """
    Whether an object is a collection.
    
    Args:
        obj: an object to test.
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


def repr_index(index: None |
                      PreSlicer |
                      tuple[PreSlicer | None] |
                      tuple[PreSlicer | None, PreSlicer | None] |
                      tuple[PreSlicer | None, PreSlicer | None, PreSlicer | None]) \
        -> str:
    """Get a short string representation of a sub-setting index."""
    if not isinstance(index, tuple):
        index = (index,)
    
    repr_string = f"Index of {len(index)} element{'' if len(index) == 1 else 's'} : "

    for element in index:
        repr_string += f"\n  \u2022 {repr_array(element) if isCollection(element) else element}"

    return repr_string

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
