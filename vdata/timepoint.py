# coding: utf-8
# Created on 01/04/2022 09:39
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Iterator, Literal, Sequence, cast, overload

import numpy as np
import numpy.typing as npt
from numpy import _NoValue as _NP_NOVALUE  # type: ignore[attr-defined]

from vdata._meta import PrettyRepr
from vdata.names import Number

if TYPE_CHECKING:
    from vdata._typing import NumberType

# ====================================================
# code
_UNIT_TYPE = Literal['s', 'm', 'h', 'D', 'M', 'Y']

_TIMEPOINT_UNITS = {'s': 'seconds',
                    'm': 'minutes',
                    'h': 'hours',
                    'D': 'days',
                    'M': 'months',
                    'Y': 'years'}

_SECONDS_IN_UNIT = {'s': 1,
                    'm': 60,
                    'h': 3_600,
                    'D': 86_400,
                    'M': 2_592_000,
                    'Y': 31_536_000}


class Unit:
    """Simple class for storing a time point's unit."""
    __slots__ = 'value'

    units = ('s', 'm', 'h', 'D', 'M', 'Y')
    units_order = {
        's': 0,
        'm': 1,
        'h': 2,
        'D': 3,
        'M': 4,
        'Y': 5
    }

    # region magic methods
    def __init__(self,
                 value: Unit | _UNIT_TYPE):
        """
        Args:
            value: a unit's value. It can be :
                - a string representing the unit, in ['s', 'm', 'h', 'D', 'M', 'Y'].
                - a Unit
        """
        if isinstance(value, Unit):
            self.value: _UNIT_TYPE = value.value

        elif value in Unit.units:
            self.value = value

        else:
            raise ValueError(f"Invalid unit '{value}', should be in {Unit.units}.")

    def __repr__(self) -> str:
        """
        A string representation of the unit as a full word.
        :return: a string representation of the unit as a full word.
        """
        return _TIMEPOINT_UNITS[self.value]

    def __str__(self) -> _UNIT_TYPE:
        return self.value

    def __gt__(self,
               other: 'Unit') -> bool:
        """
        Compare units with 'greater than'.
        """
        return Unit.units_order[self.value] > Unit.units_order[other.value]

    def __lt__(self,
               other: 'Unit') -> bool:
        """
        Compare units with 'lesser than'.
        """
        return Unit.units_order[self.value] < Unit.units_order[other.value]

    def __eq__(self,
               other: object) -> bool:
        """
        Compare units with 'equal'.
        """
        if not isinstance(other, Unit):
            raise ValueError('Not a Unit.')

        return self.value == other.value

    def __ge__(self,
               other: 'Unit') -> bool:
        """
        Compare units with 'greater or equal'.
        """
        return Unit.units_order[self.value] >= Unit.units_order[other.value]

    def __le__(self,
               other: 'Unit') -> bool:
        """
        Compare units with 'lesser or equal'.
        """
        return Unit.units_order[self.value] <= Unit.units_order[other.value]

    # endregion

class TimePoint(metaclass=PrettyRepr):
    """Simple class for storing a single time point, with its value and unit."""
    
    __slots__ = 'value', 'unit'

    # region magic methods
    def __init__(self,
                 value: TimePoint | NumberType | str,
                 unit: Unit | _UNIT_TYPE | None = None):
        """
        Args:
            value: a time-point's value. It can be :
                - a number
                - a string representing a time-point with format "<value><unit>" where <unit> is a single letter in
                    ('s', 'm', 'h', 'D', 'M', 'Y') i.e. (seconds, minutes, hours, Days, Months, Years).
                - a TimePoint
            unit: an Optional string representing a unit, in ('s', 'm', 'h', 'D', 'M', 'Y').
                /!\\ Overrides the unit defined in 'value' if 'value' is a string or a TimePoint.
        """
        if isinstance(value, bytes):
            value = value.decode()

        if isinstance(value, TimePoint):
            self.value: float = value.value
            self.unit: Unit = value.unit if unit is None else Unit(unit)

        elif isinstance(value, str):
            if value.endswith(Unit.units):
                self.value = float(value[:-1])
                self.unit = Unit(cast(_UNIT_TYPE, value[-1])) if unit is None else Unit(unit)

            else:
                self.value = float(value)
                self.unit = Unit(unit) if unit is not None else Unit('h')

        elif isinstance(value, Number):
            self.value = float(value)
            self.unit = Unit(unit) if unit is not None else Unit('h')

        else:
            raise ValueError(f"Invalid value '{value}' with type '{type(value)}'.")

    def __repr__(self) -> str:
        """
        A string representation of this time point.
        :return: a string representation of this time point.
        """
        return f"{self.value} {repr(self.unit)}"

    def __str__(self) -> str:
        """
        A short string representation where the unit is represented by a single character.
        """
        return f"{self.value}{str(self.unit)}"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __gt__(self, other: TimePoint) -> bool:
        """
        Compare units with 'greater than'.
        """
        value_self = self.value_as('s')
        value_other = other.value_as('s')

        return value_self > value_other

    def __lt__(self, other: TimePoint) -> bool:
        """
        Compare units with 'lesser than'.
        """
        value_self = self.value_as('s')
        value_other = other.value_as('s')

        return value_self < value_other

    def __eq__(self, other: object) -> bool:
        """
        Compare units with 'equal'.
        """
        if not isinstance(other, TimePoint):
            if not isinstance(other, (Number, str)):
                return False

            other = TimePoint(other)

        return self.value_as('s') == other.value_as('s')

    def __ge__(self, other: TimePoint) -> bool:
        """
        Compare units with 'greater or equal'.
        """
        return self > other or self == other

    def __le__(self, other: TimePoint) -> bool:
        """
        Compare units with 'lesser or equal'.
        """
        return self < other or self == other
    
    def __add__(self, other: TimePoint | NumberType) -> TimePoint:
        if isinstance(other, Number):
            return TimePoint(self.value + other, self.unit)
        
        value = self.value_as('s') + other.value_as('s')
        
        if value < _SECONDS_IN_UNIT['m']:
            return TimePoint(value, 's')

        elif value < _SECONDS_IN_UNIT['h']:
            return TimePoint(value / _SECONDS_IN_UNIT['m'], 'm')

        elif value < _SECONDS_IN_UNIT['D']:
            return TimePoint(value / _SECONDS_IN_UNIT['h'], 'h')

        elif value < _SECONDS_IN_UNIT['M']:
            return TimePoint(value / _SECONDS_IN_UNIT['D'], 'D')

        elif value < _SECONDS_IN_UNIT['Y']:
            return TimePoint(value / _SECONDS_IN_UNIT['M'], 'M')

        return TimePoint(value / _SECONDS_IN_UNIT['Y'], 'Y')
    
    def __truediv__(self, other: int) -> TimePoint:
        return TimePoint(self.value / other, unit=self.unit)
    
    # endregion
    
    # region methods
    def round(self,
              decimals: int = 0) -> TimePoint:
        """
        Get a TimePoint with value rounded to a given number of decimals.
        """
        return TimePoint(value=np.round(self.value, decimals=decimals),
                         unit=self.unit)

    def value_as(self,
                 unit: _UNIT_TYPE) -> float:
        """
        Get this TimePoint has a number of <unit>.
        """
        return self.value * _SECONDS_IN_UNIT[self.unit.value] / _SECONDS_IN_UNIT[unit]

    # endregion
    
class TimePointRangeIterator:
    """Iterator over a TimePointRange."""
    
    __slots__ = '_current', '_stop', '_step'

    # region magic methods
    def __init__(self,
                 start: TimePoint,
                 stop: TimePoint,
                 step: TimePoint):
        if start.unit != step.unit:
            raise ValueError("Cannot create TimePointRangeIterator if start and step time-points' units are different")

        self._current = start
        self._stop = stop
        self._step = step

    def __iter__(self) -> TimePointRangeIterator:
        return self

    def __next__(self) -> TimePoint:
        if self._current >= self._stop:
            raise StopIteration

        self._current = TimePoint(value=self._current.value + self._step.value, unit=self._current.unit)
        return self._current

    # endregion

class TimePointRange:
    """Range of TimePoints."""
    
    __slots__ = '_start', '_stop', '_step'

    # region magic methods
    def __init__(self,
                 start: str | NumberType | TimePoint,
                 stop: str | NumberType | TimePoint,
                 step: str | NumberType | TimePoint | None = None):
        self._start = TimePoint(start)
        self._stop = TimePoint(stop)
        self._step = TimePoint(value=1, unit=self._start.unit) if step is None else TimePoint(step)

        if self._start.unit != self._step.unit:
            raise ValueError("Cannot create TimePointRange if start and step time-points' units are different")

    def __iter__(self) -> TimePointRangeIterator:
        return TimePointRangeIterator(self._start, self._stop, self._step)

    # endregion


def _no_0_dim(arr: TimePointArray) -> TimePointArray | TimePoint:
    if arr.ndim == 0:
        return arr[()]
    return arr

class TimePointArray(np.ndarray[Any, Any], Sequence[TimePoint]):
    
    # region magic methods
    def __new__(cls, arr: Collection[TimePoint]) -> TimePointArray:
        return np.asarray([TimePoint(tp) for tp in arr]).view(cls)
    
    def __iter__(self) -> Iterator[TimePoint]:
        return cast(Iterator[TimePoint], super().__iter__())
    
    @overload                                                           # type: ignore[override]
    def __getitem__(self, key: int | np.int_ | tuple[int | np.int_, ...]) -> TimePoint: ...
    @overload
    def __getitem__(self, key: slice | list[int | np.int_] | npt.NDArray[np.int_ | np.bool_]) -> TimePointArray: ...
    def __getitem__(self, key: Any) -> TimePoint | TimePointArray:
        return cast(TimePoint | TimePointArray, super().__getitem__(key))
    
    # endregion
    
    # region methods
    def mean(self,      # type: ignore[override]
             axis: int | tuple[int, ...] | None = None,
             dtype: npt.DTypeLike | None = None,
             out: npt.NDArray[Any] | None = None,
             keepdims: bool = False,
             *, 
             where: bool | npt.NDArray[np.bool_] = True) -> TimePointArray | TimePoint:
        return _no_0_dim(super().mean(axis, dtype, out, keepdims, where=where))        # type: ignore[arg-type, misc]
    
    def min(self,       # type: ignore[override]
            axis: int | tuple[int, ...] | None = None,
            out: npt.NDArray[Any] | None = None,
            keepdims: bool = False,
            initial: Any = _NP_NOVALUE,
            where: bool | npt.NDArray[np.bool_] = True) -> TimePointArray | TimePoint:
        return _no_0_dim(super().min(axis, out, keepdims, initial, where))     # type: ignore[arg-type]
        
    def max(self,       # type: ignore[override]
            axis: int | tuple[int, ...] | None = None,
            out: npt.NDArray[Any] | None = None,
            keepdims: bool = False,
            initial: Any = _NP_NOVALUE,
            where: bool | npt.NDArray[np.bool_] = True) -> TimePointArray | TimePoint:
        return _no_0_dim(super().max(axis, out, keepdims, initial, where))     # type: ignore[arg-type]
    
    
    # endregion
