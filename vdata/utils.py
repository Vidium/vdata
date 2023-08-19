from __future__ import annotations

from itertools import islice
from math import ceil, floor
from typing import TYPE_CHECKING, Any, Collection, Mapping, TypeGuard, TypeVar

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

from vdata.array_view import NDArrayView

if TYPE_CHECKING:
    from vdata._typing import PreSlicer

_V = TypeVar("_V")

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


def are_equal(obj1: Any, obj2: Any) -> bool:
    if isinstance(obj1, (np.ndarray, ch.H5Array, NDArrayView)):
        if isinstance(obj2, (np.ndarray, ch.H5Array, NDArrayView)):
            return np.array_equal(obj1[:], obj2[:])

        return False

    return bool(obj1 == obj2)


def spacer(nb: int) -> str:
    return "  " * (nb - 1) + "  " + "\u21B3" + " " if nb else ""


def obj_as_str(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return arr.astype(str) if arr.dtype == object else arr


# endregion


# region Representation --------------------------------------------------------------
def repr_array(arr: Any, /, *, n_max: int = 4, print_length: bool = True) -> str:
    """Get a short string representation of an array."""
    if isinstance(arr, slice) or arr is Ellipsis or not isCollection(arr):
        return str(arr)

    arr = list(arr)

    if len(arr) <= n_max:
        if not print_length:
            return str(arr)
        return f"{str(arr)} ({len(arr)} value{'' if len(arr) == 1 else 's'} long)"

    repr_ = (
        "["
        + " ".join((str(e) for e in islice(arr, 0, ceil(n_max / 2))))
        + " ... "
        + " ".join((str(e) for e in islice(arr, len(arr) - floor(n_max / 2), None)))
        + "]"
    )

    if not print_length:
        return repr_
    return f"{repr_} ({len(arr)} values long)"


def repr_index(
    index: None
    | PreSlicer
    | tuple[PreSlicer | None]
    | tuple[PreSlicer | None, PreSlicer | None]
    | tuple[PreSlicer | None, PreSlicer | None, PreSlicer | None]
) -> str:
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
