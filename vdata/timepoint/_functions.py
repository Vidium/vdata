from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(np_function: Callable[..., Any]) -> Callable[..., Any]:
    """Register an __array_function__ implementation for H5Array objects."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.equal)
def _equal(
    x1: Any,
    x2: Any,
    /,
    out: npt.NDArray[Any] | tuple[npt.NDArray[Any]] | None = None,
    *,
    where: bool | npt.NDArray[np.bool_] = True,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    order: Literal["K", "C", "F", "A"] = "K",
    dtype: npt.DTypeLike | None = None,
) -> Any:
    x1 = tp.as_timepointarray(x1)
    x2 = tp.as_timepointarray(x2)

    if x1.unit != x2.unit:
        return np.zeros(shape=np.broadcast_shapes(x1.shape, x2.shape), dtype=bool)

    return np.equal(np.array(x1), np.array(x2), out=out, where=where, casting=casting, order=order, dtype=dtype)


@implements(np.not_equal)
def _not_equal(
    x1: npt.NDArray[Any] | tp.TimePointArray,
    x2: npt.NDArray[Any] | tp.TimePointArray,
    /,
    out: npt.NDArray[Any] | tuple[npt.NDArray[Any]] | None = None,
    *,
    where: bool | npt.NDArray[np.bool_] = True,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    order: Literal["K", "C", "F", "A"] = "K",
    dtype: npt.DTypeLike | None = None,
) -> Any:
    return ~_equal(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype)


@implements(np.in1d)
def _in1d(
    ar1: npt.NDArray[Any] | tp.TimePointArray,
    ar2: npt.NDArray[Any] | tp.TimePointArray,
    assume_unique: bool = False,
    invert: bool = False,
) -> Any:
    ar1 = tp.as_timepointarray(ar1)
    ar2 = tp.as_timepointarray(ar2)

    if ar1.unit != ar2.unit:
        return np.zeros(shape=ar1.shape, dtype=bool)

    return np.in1d(np.array(ar1), np.array(ar2), assume_unique=assume_unique, invert=invert)
