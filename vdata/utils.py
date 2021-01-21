# coding: utf-8
# Created on 21/01/2021 11:21
# Author : matteo

# ====================================================
# imports
import numpy as np
from typing import Optional, Tuple, Union, Any

from . import NameUtils
from ._IO.errors import VValueError, VTypeError


# ====================================================
# code
time_point_units = {None: '(no unit)',
                    's': 'seconds',
                    'm': 'minutes',
                    'h': 'hours',
                    'D': 'days',
                    'M': 'months',
                    'Y': 'years'}


def get_value(v: Any) -> Union[str, int, float]:
    """
    If possible, get the int or float value of the passed object.
    :param v: an object for which to try to get the value.
    :return: the object's value (int or float) or the object itself.
    """
    try:
        return eval(str(v))

    except (NameError, SyntaxError):
        return str(v)


class Unit:
    """
    Simple class for storing a time point's unit.
    """
    _units = list(time_point_units.keys())

    def __init__(self, unit: Optional[str]):
        """
        :param unit: a string representing the unit, in ['s', 'm', 'h', 'D', 'M', 'Y'].
        """
        if unit not in Unit._units:
            raise VValueError(f"Invalid unit '{unit}', should be in {Unit._units}.")

        self.unit = unit

    def __repr__(self) -> str:
        """
        A string representation of the unit as a full word.
        :return: a string representation of the unit as a full word.
        """
        return time_point_units[self.unit]

    def __gt__(self, other):
        """
        Compare units with 'greater than'.
        """
        return Unit._units.index(self.unit) > Unit._units.index(other.unit)

    def __lt__(self, other):
        """
        Compare units with 'lesser than'.
        """
        return Unit._units.index(self.unit) < Unit._units.index(other.unit)

    def __ge__(self, other):
        """
        Compare units with 'greater or equal'.
        """
        return Unit._units.index(self.unit) >= Unit._units.index(other.unit)

    def __le__(self, other):
        """
        Compare units with 'lesser or equal'.
        """
        return Unit._units.index(self.unit) <= Unit._units.index(other.unit)

    def __eq__(self, other):
        """
        Compare units with 'equal'.
        """
        return Unit._units.index(self.unit) == Unit._units.index(other.unit)


class TimePoint:
    """
    Simple class for storing a single time point, with its value and unit.
    """

    def __init__(self, time_point: Union['NameUtils.DType', 'TimePoint']):
        """
        :param time_point: a time point's value. It can be an int or a float, or a string with format "<value><unit>"
            where <unit> is a single letter in s, m, h, D, M, Y (seconds, minutes, hours, Days, Months, Years).
        """
        if isinstance(time_point, TimePoint):
            self.value, self.unit = time_point.value, time_point.unit

        else:
            self.value, self.unit = self.__parse(time_point)

    @staticmethod
    def __parse(time_point: 'NameUtils.DType') -> Tuple['NameUtils.DType', Optional[Unit]]:
        """
        Get time point's value and unit.

        :param time_point: a time point's value given by the user.
        :return: tuple of value and unit.
        """
        if isinstance(time_point, (int, float, np.int, np.float)):
            return time_point, Unit(None)

        elif isinstance(time_point, (str, np.str)):
            v = get_value(time_point)

            if isinstance(v, str):
                if len(time_point) > 1:
                    # try to get unit
                    v, u = get_value(time_point[:-1]), time_point[-1]

                    if not isinstance(v, (int, float, np.int, np.float)):
                        raise VValueError(f"Invalid time point value '{time_point}'")

                    else:
                        return v, Unit(u)

                elif time_point == '*':
                    return '*', Unit(None)

                else:
                    raise VValueError(f"Invalid time point value '{time_point}'")

            else:
                return v, Unit(None)

        else:
            raise VTypeError(f"Invalid type '{type(time_point)}' for TimePoint.")

    def __repr__(self) -> str:
        """
        A string representation of this time point.
        :return: a string representation of this time point.
        """
        return f"{self.value} {self.unit}"

    def __gt__(self, other):
        """
        Compare units with 'greater than'.
        """
        return self.unit > other.unit or (self.unit == other.unit and self.value != '*' and self.value > other.value)

    def __lt__(self, other):
        """
        Compare units with 'lesser than'.
        """
        return self.unit < other.unit or (self.unit == other.unit and self.value != '*' and self.value < other.value)

    def __ge__(self, other):
        """
        Compare units with 'greater or equal'.
        """
        return self > other or self == other

    def __le__(self, other):
        """
        Compare units with 'lesser or equal'.
        """
        return self < other or self == other

    def __eq__(self, other):
        """
        Compare units with 'equal'.
        """
        return self.unit == other.unit and self.value == other.value

    def __hash__(self):
        return hash(repr(self))
