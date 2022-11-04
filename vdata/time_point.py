# coding: utf-8
# Created on 01/04/2022 09:39
# Author : matteo

# ====================================================
# imports
import numpy as np
from numbers import Number

from typing import Union, Literal, cast, Sequence

from vdata._meta import PrettyRepr

# ====================================================
# code
UNIT_TYPE = Literal['s', 'm', 'h', 'D', 'M', 'Y']

time_point_units = {'s': 'seconds',
                    'm': 'minutes',
                    'h': 'hours',
                    'D': 'days',
                    'M': 'months',
                    'Y': 'years'}

time_point_units_seconds = {'s': 1,
                            'm': 60,
                            'h': 3600,
                            'D': 86400,
                            'M': 2592000,
                            'Y': 31104000}


class Unit:
    """
    Simple class for storing a time point's unit.
    """
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

    def __init__(self,
                 value: Union['Unit', UNIT_TYPE]):
        """
        Args:
            value: a unit's value. It can be :
                - a string representing the unit, in ['s', 'm', 'h', 'D', 'M', 'Y'].
                - a Unit
        """
        if isinstance(value, Unit):
            self.value: UNIT_TYPE = value.value

        elif value in Unit.units:
            self.value = value

        else:
            raise ValueError(f"Invalid unit '{value}', should be in {Unit.units}.")

    def __repr__(self) -> str:
        """
        A string representation of the unit as a full word.
        :return: a string representation of the unit as a full word.
        """
        return time_point_units[self.value]

    def __str__(self) -> UNIT_TYPE:
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


class TimePoint(metaclass=PrettyRepr):
    """
    Simple class for storing a single time point, with its value and unit.
    """
    __slots__ = 'value', 'unit'

    def __init__(self,
                 value: Union['TimePoint', Number, np.number, str],
                 unit: Union[None, Unit, UNIT_TYPE] = None):
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
        if isinstance(value, TimePoint):
            self.value: float = value.value
            self.unit: Unit = value.unit if unit is None else Unit(unit)

        elif isinstance(value, str):
            if value.endswith(Unit.units):
                self.value = float(value[:-1])
                self.unit = Unit(cast(UNIT_TYPE, value[-1])) if unit is None else Unit(unit)

            else:
                self.value = float(value)
                self.unit = Unit(unit) if unit is not None else Unit('h')

        elif isinstance(value, (Number, np.number)):
            self.value = float(value)                                                        # type: ignore
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

    def round(self,
              decimals=0) -> 'TimePoint':
        """
        Get a TimePoint with value rounded to a given number of decimals.
        """
        return TimePoint(value=np.round(self.value, decimals=decimals),
                         unit=self.unit)

    def get_value_as(self,
                     unit: UNIT_TYPE) -> float:
        """
        Get this TimePoint has a number of <unit>.
        """
        return self.value * time_point_units_seconds[self.unit.value] / time_point_units_seconds[unit]

    def __hash__(self) -> int:
        return hash(repr(self))

    def __gt__(self,
               other: 'TimePoint') -> bool:
        """
        Compare units with 'greater than'.
        """
        value_self = self.get_value_as('s')
        value_other = other.get_value_as('s')

        return value_self > value_other

    def __lt__(self,
               other: 'TimePoint') -> bool:
        """
        Compare units with 'lesser than'.
        """
        value_self = self.get_value_as('s')
        value_other = other.get_value_as('s')

        return value_self < value_other

    def __eq__(self,
               other: object) -> bool:
        """
        Compare units with 'equal'.
        """
        if not isinstance(other, TimePoint):
            if not isinstance(other, (Number, np.number, str)):
                return False

            other = TimePoint(other)

        return self.get_value_as('s') == other.get_value_as('s')

    def __ge__(self,
               other: 'TimePoint') -> bool:
        """
        Compare units with 'greater or equal'.
        """
        return self > other or self == other

    def __le__(self,
               other: 'TimePoint') -> bool:
        """
        Compare units with 'lesser or equal'.
        """
        return self < other or self == other


class TimePointRangeIterator:

    __slots__ = '_current', '_stop', '_step'

    def __init__(self,
                 start: TimePoint,
                 stop: TimePoint,
                 step: TimePoint):
        if start.unit != step.unit:
            raise ValueError("Cannot create TimePointRangeIterator if start and step time-points' units are different")

        self._current = start
        self._stop = stop
        self._step = step

    def __iter__(self) -> 'TimePointRangeIterator':
        return self

    def __next__(self) -> TimePoint:
        if self._current >= self._stop:
            raise StopIteration

        self._current = TimePoint(value=self._current.value + self._step.value, unit=self._current.unit)
        return self._current


class TimePointRange:

    __slots__ = '_start', '_stop', '_step'

    def __init__(self,
                 start: Union[str, Number, np.number, TimePoint],
                 stop: Union[str, Number, np.number, TimePoint],
                 step: Union[str, Number, np.number, TimePoint] = None):
        self._start = TimePoint(start)
        self._stop = TimePoint(stop)
        self._step = TimePoint(value=1, unit=self._start.unit) if step is None else TimePoint(step)

        if self._start.unit != self._step.unit:
            raise ValueError("Cannot create TimePointRange if start and step time-points' units are different")

    def __iter__(self) -> TimePointRangeIterator:
        return TimePointRangeIterator(self._start, self._stop, self._step)


def mean(timepoints: Sequence[TimePoint]):
    mean_ = float(np.mean([tp.get_value_as('s') for tp in timepoints]))

    if mean_ < 60:
        return TimePoint(mean_, 's')                                                        # type: ignore

    elif mean_ < 3600:                   # 60 * 60
        return TimePoint(mean_, 'm')                                                        # type: ignore

    elif mean_ < 86_400:                 # 60 * 60 * 24
        return TimePoint(mean_, 'h')                                                        # type: ignore

    elif mean_ < 2_592_000:              # 60 * 60 * 24 * 30
        return TimePoint(mean_, 'D')                                                        # type: ignore

    elif mean_ < 31_536_000:             # 60 * 60 * 24 * 365
        return TimePoint(mean_, 'M')                                                        # type: ignore

    return TimePoint(mean_, 'Y')                                                            # type: ignore
