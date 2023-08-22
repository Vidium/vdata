from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp
from vdata.array_view import NDArrayView

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
    *,
    kind: Literal["sort", "table"] | None = None,
) -> Any:
    tp_arr_1 = tp.as_timepointarray(ar1)
    tp_arr_2 = tp.as_timepointarray(ar2)

    if tp_arr_1.unit != tp_arr_2.unit:
        return np.zeros(shape=tp_arr_1.shape, dtype=bool)

    return np.in1d(
        np.array(tp_arr_1), np.array(tp_arr_2), assume_unique=assume_unique, invert=invert, kind=kind
    )  # type: ignore[call-arg]


@implements(np.isin)
def _isin(
    element: npt.NDArray[Any] | tp.TimePointArray,
    test_elements: npt.NDArray[Any] | tp.TimePointArray,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: Literal["sort", "table"] | None = None,
) -> Any:
    tp_arr_element = tp.as_timepointarray(element)
    tp_arr_test_elements = tp.as_timepointarray(test_elements)

    if tp_arr_element.unit != tp_arr_test_elements.unit:
        return np.zeros(shape=tp_arr_element.shape, dtype=bool)

    return np.isin(
        np.array(tp_arr_element), np.array(tp_arr_test_elements), assume_unique=assume_unique, invert=invert, kind=kind
    )  # type: ignore[call-arg]


@implements(np.repeat)
def repeat(
    a: tp.TimePointArray, repeats: int | npt.NDArray[np.int_], axis: int | None = None
) -> tp.TimePointArray | NDArrayView[tp.TimePoint]:
    return tp.as_timepointarray(np.repeat(np.array(a), repeats, axis=axis), unit=a.unit)
