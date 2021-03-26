# coding: utf-8
# Created on 05/03/2021 16:10
# Author : matteo

# ====================================================
# imports
import numpy as np
from typing import Optional, Union, cast, Tuple, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    from .NameUtils import DType

from .utils import get_value
from ._IO import VValueError, VTypeError

# ====================================================
# code
time_point_units = {None: '(no unit)',
                    's': 'seconds',
                    'm': 'minutes',
                    'h': 'hours',
                    'D': 'days',
                    'M': 'months',
                    'Y': 'years'}

time_point_units_seconds = {None: 1,
                            's': 1,
                            'm': 60,
                            'h': 3600,
                            'D': 86400,
                            'M': 2592000,
                            'Y': 31104000}

_units = (None, 's', 'm', 'h', 'D', 'M', 'Y')


class Unit:
    """
    Simple class for storing a time point's unit.
    """
    _units_order = {None: 0,
                    's': 1,
                    'm': 2,
                    'h': 3,
                    'D': 4,
                    'M': 5,
                    'Y': 6}

    def __init__(self, value: Optional[str]):
        """
        :param value: a string representing the unit, in [None, 's', 'm', 'h', 'D', 'M', 'Y'].
        """
        if value not in _units:
            raise VValueError(f"Invalid unit '{value}', should be in {_units}.")

        self.value = value

    def __repr__(self) -> str:
        """
        A string representation of the unit as a full word.
        :return: a string representation of the unit as a full word.
        """
        return time_point_units[self.value]

    def __gt__(self, other: 'Unit') -> bool:
        """
        Compare units with 'greater than'.
        """
        return Unit._units_order[self.value] > Unit._units_order[other.value]

    def __lt__(self, other: 'Unit') -> bool:
        """
        Compare units with 'lesser than'.
        """
        return Unit._units_order[self.value] < Unit._units_order[other.value]

    def __ge__(self, other: 'Unit') -> bool:
        """
        Compare units with 'greater or equal'.
        """
        return Unit._units_order[self.value] >= Unit._units_order[other.value]

    def __le__(self, other: 'Unit') -> bool:
        """
        Compare units with 'lesser or equal'.
        """
        return Unit._units_order[self.value] <= Unit._units_order[other.value]

    def __eq__(self, other: object) -> bool:
        """
        Compare units with 'equal'.
        """
        if not isinstance(other, Unit):
            raise VValueError('Not a Unit.')

        return self.value == other.value


class TimePoint:
    """
    Simple class for storing a single time point, with its value and unit.
    """

    def __init__(self, time_point: Union['DType', 'TimePoint']):
        """
        :param time_point: a time point's value. It can be an int or a float, or a string with format "<value><unit>"
            where <unit> is a single letter in s, m, h, D, M, Y (seconds, minutes, hours, Days, Months, Years).
        """
        if isinstance(time_point, TimePoint):
            self.value: 'DType' = time_point.value
            self.unit: Unit = time_point.unit

        else:
            self.value, self.unit = self.__parse(time_point)

    @staticmethod
    def __parse(time_point: Union[str, 'DType']) -> Tuple[float, Unit]:
        """
        Get time point's value and unit.

        :param time_point: a time point's value given by the user.
        :return: tuple of value and unit.
        """
        _type_time_point = type(time_point)

        if _type_time_point in (int, float, np.int_, np.float_):
            return float(time_point), Unit(None)

        elif _type_time_point in (str, np.str_):
            time_point = cast(str, time_point)
            if time_point.endswith(_units[1:]) and len(time_point) > 1:
                # try to get unit
                v, u = get_value(time_point[:-1]), time_point[-1]

                if not isinstance(v, (int, float, np.int, np.float)):
                    raise VValueError(f"Invalid time point value '{time_point}'")

                else:
                    return float(v), Unit(u)

            else:
                v = get_value(time_point)
                if isinstance(v, str):
                    raise VValueError(f"Invalid time point value '{time_point}'")

                else:
                    return float(v), Unit(None)

        else:
            raise VTypeError(f"Invalid type '{type(time_point)}' for TimePoint.")

    def __repr__(self) -> str:
        """
        A string representation of this time point.
        :return: a string representation of this time point.
        """
        return f"{self.value} {self.unit}"

    def __str__(self) -> str:
        """
        A short string representation where the unit is represented by a single character.
        """
        return f"{self.value}" \
               f"{self.unit.value if self.unit.value is not None else ''}"

    def get_unit_value(self, unit: Literal['s', 'm', 'h', 'D', 'M', 'Y']) -> float:
        """
        Get this TimePoint has a number of <unit>.
        """
        return self.value * time_point_units_seconds[self.unit.value] / time_point_units_seconds[unit]

    def __gt__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'greater than'.
        """
        vs = self.get_unit_value('s')
        vo = other.get_unit_value('s')
        return vs > vo or (vs == vo and _units.index(self.unit.value) > _units.index(other.unit.value))

    def __lt__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'lesser than'.
        """
        vs = self.get_unit_value('s')
        vo = other.get_unit_value('s')
        return vs < vo or (vs == vo and _units.index(self.unit.value) < _units.index(other.unit.value))

    def __ge__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'greater or equal'.
        """
        return self > other or self == other

    def __le__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'lesser or equal'.
        """
        return self < other or self == other

    def __eq__(self, other: object) -> bool:
        """
        Compare units with 'equal'.
        """
        if not isinstance(other, TimePoint):
            return False
        return self.get_unit_value('s') == other.get_unit_value('s')

    def __hash__(self) -> int:
        return hash(repr(self))
