from __future__ import annotations

import numpy as np

from vdata._meta import PrettyRepr
from vdata.timepoint._typing import _TIME_UNIT

_TIME_UNIT_STR = {"s": "second", "m": "minute", "h": "hour", "D": "day", "M": "month", "Y": "year"}
_TIME_UNIT_ORDER = {"s": 0, "m": 1, "h": 2, "D": 3, "M": 4, "Y": 5}
_SECONDS_IN_UNIT = {"s": 1, "m": 60, "h": 3_600, "D": 86_400, "M": 2_592_000, "Y": 31_536_000}


class TimePoint(metaclass=PrettyRepr):

    __slots__ = "value", "_unit"

    # region magic methods
    def __init__(
        self, value: TimePoint | int | float | np.int_ | np.float_ | bool | str | bytes, unit: _TIME_UNIT | None = None
    ) -> None:
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
            self._unit: _TIME_UNIT = value._unit if unit is None else unit

        elif isinstance(value, (int, float, np.int_, np.float_, bool)):
            self.value = float(value)
            self._unit = "h" if unit is None else unit

        elif isinstance(value, str):
            if value.endswith(("s", "m", "h", "D", "M", "Y")):
                self.value = float(value[:-1])
                self._unit = value[-1] if unit is None else unit  # type: ignore[assignment]

            else:
                self.value = float(value)
                self._unit = "h" if unit is None else unit

        else:
            raise ValueError(f"Invalid value '{value}' with type '{type(value)}'.")

    def __repr__(self) -> str:
        return f"{self.value} {_TIME_UNIT_STR[self._unit]}{'' if self.value == 1 else 's'}"

    def __str__(self) -> str:
        return f"{self.value}{self._unit}"

    def __format__(self, format_spec: str) -> str:
        return f"{format(self.value, format_spec)}{self._unit}"

    def __float__(self) -> float:
        return self.value

    def __hash__(self) -> int:
        return hash((self.value, self._unit))

    def __gt__(self, other: TimePoint) -> bool:
        return self.value_as("s") > other.value_as("s")

    def __lt__(self, other: TimePoint) -> bool:
        return self.value_as("s") < other.value_as("s")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimePoint):
            return self.value_as("s") == other.value_as("s")

        if isinstance(other, (int, float, np.int_, np.float_, str)):
            other = TimePoint(other)

        return False

    def __ge__(self, other: TimePoint) -> bool:
        return self.value_as("s") >= other.value_as("s")

    def __le__(self, other: TimePoint) -> bool:
        return self.value_as("s") <= other.value_as("s")

    def __add__(self, other: TimePoint | int | float | np.int_ | np.float_) -> TimePoint:
        if isinstance(other, (int, float, np.int_, np.float_)):
            return TimePoint(self.value + float(other), self.unit)

        if _TIME_UNIT_ORDER[self._unit] > _TIME_UNIT_ORDER[other._unit]:
            return TimePoint(self.value + other.value_as(self._unit), self._unit)

        return TimePoint(self.value_as(other._unit) + other.value, other._unit)

    def __mul__(self, other: int | float | np.int_ | np.float_) -> TimePoint:
        return TimePoint(self.value * other, unit=self.unit)

    def __truediv__(self, other: int | float | np.int_ | np.float_) -> TimePoint:
        return TimePoint(self.value / other, unit=self.unit)

    # endregion

    # region attributes
    @property
    def unit(self) -> _TIME_UNIT:
        return self._unit

    # endregion

    # region methods
    def round(self, decimals: int = 0) -> TimePoint:
        """Get a TimePoint with value rounded to a given number of decimals."""
        return TimePoint(value=np.round(self.value, decimals=decimals), unit=self.unit)

    def value_as(self, unit: _TIME_UNIT) -> float:
        """Get this TimePoint has a number of <unit>."""
        if unit == self._unit:
            return self.value

        return self.value * _SECONDS_IN_UNIT[self._unit] / _SECONDS_IN_UNIT[unit]

    # endregion
