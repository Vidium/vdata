from __future__ import annotations

import re
from functools import partial
from types import EllipsisType
from typing import Any, Callable, Sequence, SupportsIndex, TypeVar, overload

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
from numpy import _NoValue as NoValue  # type: ignore[attr-defined]
from numpy._typing import _ArrayLikeInt_co

from vdata.timepoint._functions import HANDLED_FUNCTIONS
from vdata.timepoint._typing import _TIME_UNIT
from vdata.timepoint.range import TimePointRange
from vdata.timepoint.timepoint import _TIME_UNIT_ORDER, TimePoint
from vdata.utils import isCollection

_T_NP = TypeVar('_T_NP', bound=np.generic)


def _add_unit(match: re.Match, unit: _TIME_UNIT) -> str:
    number = match.group(0)
    if number.endswith('.'):
        number += '0'
    return number + unit


class TimePointArray(np.ndarray[Any, np.float64], Sequence[TimePoint]):
    
    # region magic methods
    def __new__(cls, 
                arr: Sequence[np.number], 
                /, *, 
                unit: _TIME_UNIT | None = None) -> TimePointArray:
        arr = np.asarray(arr, dtype=np.float64).view(cls)
        arr._unit = unit or 'h'
        return arr
    
    @classmethod
    def from_objects(cls, 
                     arr: Any, 
                     /, *, 
                     unit: _TIME_UNIT | None = None) -> TimePointArray:
        arr = np.atleast_1d(arr)
                   
        if unit is not None:
            return TimePointArray([TimePoint(tp, unit=unit).value_as(unit) for tp in arr],
                                  unit=unit)
                  
        _timepoint_list = [TimePoint(tp) for tp in arr]
        _largest_unit = sorted(np.unique([tp.unit for tp in _timepoint_list]), 
                               key=lambda u: _TIME_UNIT_ORDER[u])[0]
        
        return TimePointArray([tp.value_as(_largest_unit) for tp in _timepoint_list],
                              unit=_largest_unit)
            
    @classmethod
    def from_range(cls, range: TimePointRange) -> TimePointArray:
        return TimePointArray(np.arange(float(range.start), float(range.stop), float(range.step)),
                              unit=range.unit)
    
    @classmethod
    def __h5_read__(self, values: ch.H5Dict[Any]) -> TimePointArray:
        return TimePointArray(values['array'], unit=values.attributes['unit'])
    
    def __array_finalize__(self, obj: TimePointArray | None) -> None:
        if self.ndim == 0:
            self.shape = (1,)
            
        if self.ndim > 1:
            raise ValueError('A TimePointArray must be 1-dimensional.')
        
        if obj is not None:                 
            self._unit = getattr(obj, '_unit', None)
    
    def __repr__(self) -> str:
        if self.size:
            return re.sub(r'(\d+(\.\d*)?|\d+)', 
                          partial(_add_unit, unit=self._unit), 
                          super().__repr__())
            
        return super().__repr__()[:-1] + f', unit={self._unit})'
    
    def __str__(self) -> str:
        return re.sub(r'(\d+(\.\d*)?|\d+)', 
                        partial(_add_unit, unit=self._unit), 
                        super().__str__())
    
    @overload
    def __getitem__(self, key: SupportsIndex) -> TimePoint: ...
    @overload
    def __getitem__(self, key: tuple[()] | EllipsisType | slice | range) -> TimePointArray: ...
    @overload
    def __getitem__(self, key: _ArrayLikeInt_co) -> TimePoint | TimePointArray: ...
    def __getitem__(self, key: int | 
                               SupportsIndex |
                               tuple[()] | 
                               EllipsisType |
                               slice |
                               range |
                               _ArrayLikeInt_co) -> TimePoint | TimePointArray:
        if isinstance(key, SupportsIndex) and not isinstance(key, np.ndarray):
            return TimePoint(super().__getitem__(key), unit=self._unit)

        return super().__getitem__(key)
    
    def __array_ufunc__(self, 
                        ufunc: np.ufunc,
                        method: str,
                        *inputs: Any,
                        **kwargs: Any) -> Any:
        if method == "__call__":
            if ufunc not in HANDLED_FUNCTIONS:
                return NotImplemented

            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)

        else:
            raise NotImplementedError
        
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> npt.NDArray[Any]:
        if func not in HANDLED_FUNCTIONS:
            return super().__array_function__(func, types, args, kwargs)

        return HANDLED_FUNCTIONS[func](*args, **kwargs)
        
    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        ch.attributes['unit'] = self._unit
        ch.write_dataset(values, 'array', np.array(self))
        
    # endregion

    # region atrtibutes
    @property
    def unit(self) -> _TIME_UNIT:
        return self._unit
    
    # endregion
    
    # region methods
    def astype(self, dtype: npt.DTypeLike[_T_NP], copy: bool = True) -> npt.NDArray[_T_NP]:
        return np.array(self, dtype=dtype)
    
    def min(self, 
            axis: Any = None,
            out: Any = None,
            keepdims: Any = None,
            initial: int | float | NoValue = NoValue,
            where: bool | npt.NDArray[np.bool_] = True) -> TimePoint:
        return TimePoint(np.min(np.array(self), initial=initial, where=where), unit=self._unit)
    
    def max(self, 
            axis: Any = None,
            out: Any = None,
            keepdims: Any = None,
            initial: int | float | NoValue = NoValue,
            where: bool | npt.NDArray[np.bool_] = True) -> TimePoint:
        return TimePoint(np.max(np.array(self), initial=initial, where=where), unit=self._unit)
    
    def mean(self, 
             axis: Any = None,
             dtype: Any = None,
             out: Any = None,
             keepdims: Any = None,
             where: bool | npt.NDArray[np.bool_] = True) -> TimePoint:
        return TimePoint(np.mean(np.array(self), where=where), unit=self._unit)
        
    # endregion

def atleast_1d(obj: Any) -> TimePointArray:
    if isinstance(obj, Sequence[Any]):
        return TimePointArray(obj)
    
    return TimePointArray([obj])


def get_time_array(time_list: Any) -> TimePointArray:
    if not isCollection(time_list):
        time_list = [time_list]
    
    try:
        return TimePointArray(time_list)
    
    except ValueError:
        try: 
            return TimePointArray.from_objects(time_list)
            
        except ValueError as e:
            raise TypeError(f"Unexpected type '{type(time_list)}' for 'time_list', " 
                            f"should be a collection of time-points.") from e
