from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp
from vdata.array_view import NDArrayView
from vdata.timepoint._typing import _TIME_UNIT

HANDLED_FUNCTIONS: dict[np.ufunc, Callable[..., Any]] = {}


def implements(np_function: np.ufunc) -> Callable[..., Any]:
    """Register an __array_function__ implementation for H5Array objects."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def _ensure_TimePointArray_first(x1: Any, 
                                 x2: Any) -> tuple[tp.TimePointArray, Any]:
    if isinstance(x1, tp.TimePointArray):
        return x1, x2
    elif isinstance(x2, tp.TimePointArray):
        return x2, x1
    raise RuntimeError


def _get_array_and_unit(obj: Any, default_unit: _TIME_UNIT) -> tuple[npt.NDArray[Any], _TIME_UNIT]:
    if isinstance(obj, tp.TimePointArray):
        return np.array(obj), obj.unit
    
    if isinstance(obj, NDArrayView) and obj.array_type is tp.TimePointArray:
        return np.array(obj), obj.unit
    
    if isinstance(obj, tp.TimePoint):
        return np.array([obj.value]), obj.unit
    
    obj = np.atleast_1d(obj)
    
    try:
        return np.asarray(obj, dtype=np.float64), default_unit
    
    except ValueError:
        pass
    
    try:
        _tp = np.asarray([e[:-1] for e in obj], dtype=np.float64)
    
    except ValueError as e:
        raise ValueError(f"Cannot build TimePointArray from values '{obj}'.") from e
        
    _unit = np.unique([e[-1] for e in obj])
    
    if len(_unit) != 1:
        raise ValueError('Cannot build TimePointArray from values with different time units.')
    
    if _unit[0] not in ('s', 'm', 'h', 'D', 'M', 'Y'):
        raise ValueError(f"Invalid time unit '{_unit[0]}'.")
    
    return _tp, _unit[0]


@implements(np.equal)
def _equal(x1: Any, 
           x2: Any,
           /, 
           out: npt.NDArray[Any] | tuple[npt.NDArray[Any]] | None = None,
           *, 
           where: bool | npt.NDArray[np.bool_] = True,
           casting: Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe'] = 'same_kind',
           order: Literal['K', 'C', 'F', 'A'] = 'K', 
           dtype: npt.DTypeLike | None = None) -> Any:        
    x1, x2 = _ensure_TimePointArray_first(x1, x2)
    x2, x2_unit = _get_array_and_unit(x2, x1.unit)
    
    if x1.unit != x2_unit:
        return np.zeros(shape=np.broadcast_shapes(x1.shape, x2.shape), dtype=bool)    
    
    return np.equal(np.array(x1), x2, 
                    out=out, where=where, casting=casting, order=order, dtype=dtype)


@implements(np.not_equal)
def _not_equal(x1: npt.NDArray[Any] | tp.TimePointArray, 
               x2: npt.NDArray[Any] | tp.TimePointArray,
               /, 
               out: npt.NDArray[Any] | tuple[npt.NDArray[Any]] | None = None,
               *, 
               where: bool | npt.NDArray[np.bool_] = True,
               casting: Literal['no', 'equiv', 'safe', 'same_kind', 'unsafe'] = 'same_kind',
               order: Literal['K', 'C', 'F', 'A'] = 'K', 
               dtype: npt.DTypeLike | None = None) -> Any:
    return ~_equal(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype)


@implements(np.in1d)
def _in1d(ar1: npt.NDArray[Any] | tp.TimePointArray,
          ar2: npt.NDArray[Any] | tp.TimePointArray,
          assume_unique: bool = False,
          invert: bool = False) -> Any:
    x1, x2 = _ensure_TimePointArray_first(ar1, ar2)
    _, x2_unit = _get_array_and_unit(x2, x1.unit)
    
    if x1.unit != x2_unit:
        return np.zeros(shape=ar1.shape, dtype=bool)
    
    return np.in1d(np.array(ar1), np.array(ar2))
