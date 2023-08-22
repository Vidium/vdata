from __future__ import annotations

import re
from functools import partial
from types import EllipsisType
from typing import Any, Callable, Collection, Iterable, Mapping, SupportsIndex, cast, overload

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
from numpy import _NoValue as NoValue  # type: ignore[attr-defined]
from numpy._typing import _ArrayLikeInt_co

from vdata._meta import PrettyRepr
from vdata.array_view import NDArrayView
from vdata.timepoint._functions import HANDLED_FUNCTIONS
from vdata.timepoint._typing import _TIME_UNIT
from vdata.timepoint.range import TimePointRange
from vdata.timepoint.timepoint import _TIME_UNIT_ORDER, TimePoint
from vdata.utils import isCollection


def _add_unit(match: re.Match[Any], unit: _TIME_UNIT) -> str:
    number = str(match.group(0))
    if number.endswith("."):
        number += "0"
    return number + unit


class TimePointArray(np.ndarray[Any, Any], metaclass=PrettyRepr):

    # region magic methods
    def __new__(
        cls, arr: Collection[int | float | np.int_ | np.float_], /, *, unit: _TIME_UNIT | None = None
    ) -> TimePointArray:
        if isinstance(arr, TimePointArray):
            unit = arr.unit

        np_arr = np.asarray(arr, dtype=np.float64).view(cls)
        np_arr._unit = unit or "h"
        return np_arr

    @classmethod
    def __h5_read__(self, values: ch.H5Dict[Any]) -> TimePointArray:
        return TimePointArray(values["array"], unit=values.attributes["unit"])

    def __array_finalize__(self, obj: npt.NDArray[np.float_] | None) -> None:
        if self.ndim == 0:
            self.shape = (1,)

        if obj is not None:
            self._unit: _TIME_UNIT = getattr(obj, "_unit", "h")

    def __repr__(self) -> str:
        # TODO : modify for all units
        if self.size:
            return f"{type(self).__name__}({re.sub(r'h ', r'h, ', str(self))})"

        return f"{type(self).__name__}({re.sub(r'h ', r'h, ', str(self))}, unit={self._unit})"

    def __str__(self) -> str:
        return re.sub(r"(\d+(\.\d*)?|\d+)", partial(_add_unit, unit=self._unit), str(self.__array__()))

    @overload  # type: ignore[override]
    def __getitem__(self, key: SupportsIndex) -> TimePoint:
        ...

    @overload
    def __getitem__(self, key: tuple[()] | EllipsisType | slice | range) -> TimePointArray:
        ...

    @overload
    def __getitem__(self, key: _ArrayLikeInt_co) -> TimePoint | TimePointArray:
        ...

    def __getitem__(
        self, key: int | SupportsIndex | tuple[()] | EllipsisType | slice | range | _ArrayLikeInt_co
    ) -> TimePoint | TimePointArray:
        res = super().__getitem__(key)
        if isinstance(res, TimePointArray):
            return res

        return TimePoint(cast(np.float64, res), unit=self._unit)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            raise NotImplementedError

        if ufunc in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)

        _inputs = (np.array(i) if isinstance(i, TimePointArray) else i for i in inputs)
        _kwargs = {k: np.array(v) if isinstance(v, TimePointArray) else v for k, v in kwargs.items()}
        return ufunc(*_inputs, **_kwargs)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if func not in HANDLED_FUNCTIONS:
            return super().__array_function__(func, types, args, kwargs)

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __h5_write__(self, values: ch.H5Dict[Any]) -> None:
        ch.attributes["unit"] = self._unit
        ch.write_dataset(values, "array", np.array(self))

    # endregion

    # region atrtibutes
    @property
    def unit(self) -> _TIME_UNIT:
        return self._unit

    # endregion

    # region methods
    def astype(self, dtype: npt.DTypeLike, copy: bool = True) -> npt.NDArray[Any]:  # type: ignore[override]
        if np.issubdtype(dtype, str):
            return np.char.add(np.array(self, dtype=dtype), self._unit)

        return np.array(self, dtype=dtype)

    def min(  # type: ignore[override]
        self,
        axis: Any = None,
        out: Any = None,
        keepdims: Any = None,
        initial: int | float | NoValue = NoValue,
        where: bool | npt.NDArray[np.bool_] = True,
    ) -> TimePoint:
        del keepdims
        return TimePoint(np.min(np.array(self), initial=initial, where=where), unit=self._unit)

    def max(  # type: ignore[override]
        self,
        axis: Any = None,
        out: Any = None,
        keepdims: Any = None,
        initial: int | float | NoValue = NoValue,
        where: bool | npt.NDArray[np.bool_] = True,
    ) -> TimePoint:
        del keepdims
        return TimePoint(np.max(np.array(self), initial=initial, where=where), unit=self._unit)

    def mean(  # type: ignore[override]
        self,
        axis: Any = None,
        dtype: Any = None,
        out: Any = None,
        keepdims: Any = None,
        where: bool | npt.NDArray[np.bool_] = True,
    ) -> TimePoint:
        del keepdims
        return TimePoint(np.mean(np.array(self), where=where), unit=self._unit)

    def to_list(self) -> list[TimePoint]:
        return [tp for tp in self]

    # endregion


def atleast_1d(obj: Any) -> TimePointArray | NDArrayView[Any]:
    if isinstance(obj, TimePointArray) or (isinstance(obj, NDArrayView) and obj.array_type == TimePointArray):
        return obj

    if isinstance(obj, Collection):
        return TimePointArray(obj)

    return TimePointArray([obj])


def as_timepointarray(time_list: Any, /, *, unit: _TIME_UNIT | None = None) -> TimePointArray | NDArrayView[TimePoint]:
    """
    Args:
        time_list: a list for timepoints (TimePointArray, TimePointRange, object or collection of objects).
        unit: enforce a time unit. /!\ replaces any unit found in `time_list`.
    """
    if isinstance(time_list, TimePointArray) or (
        isinstance(time_list, NDArrayView) and time_list.array_type == TimePointArray
    ):
        return time_list

    if isinstance(time_list, TimePointRange):
        return TimePointArray(
            np.arange(float(time_list.start), float(time_list.stop), float(time_list.step)), unit=time_list.unit
        )

    if not isCollection(time_list):
        time_list = [time_list]
    time_list = np.array(time_list)

    if not time_list.size or np.issubdtype(time_list.dtype, np.floating) or np.issubdtype(time_list.dtype, np.integer):
        return TimePointArray(time_list, unit=unit)

    if np.issubdtype(time_list.dtype, str):
        unique_units = np.unique([e[-1] for e in time_list])

        if len(unique_units) == 1 and unique_units[0] in {"s", "m", "h", "D", "M", "Y"}:
            dtype = time_list.dtype.str[:-1] + str(int(time_list.dtype.str[-1]) - 1)
            try:
                return TimePointArray(time_list.astype(dtype), unit=unique_units[0] if unit is None else unit)

            except ValueError:
                pass

    if unit is not None:
        return TimePointArray([TimePoint(tp, unit=unit).value_as(unit) for tp in np.atleast_1d(time_list)], unit=unit)

    _timepoint_list = [TimePoint(tp) for tp in np.atleast_1d(time_list)]
    _largest_unit = sorted(np.unique([tp.unit for tp in _timepoint_list]), key=lambda u: _TIME_UNIT_ORDER[u])[0]

    try:
        return TimePointArray([tp.value_as(_largest_unit) for tp in _timepoint_list], unit=_largest_unit)

    except ValueError as e:
        raise TypeError(
            f"Unexpected type '{type(time_list)}' for 'time_list', " f"should be a collection of time-points."
        ) from e
