# coding: utf-8
# Created on 21/01/2021 11:21
# Author : matteo

# ====================================================
# imports
import builtins
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Any, Collection, List, Set, Sequence, cast

from . import NameUtils
from ._IO.errors import VValueError, VTypeError, ShapeError


# ====================================================
# code
time_point_units = {None: '(no unit)',
                    's': 'seconds',
                    'm': 'minutes',
                    'h': 'hours',
                    'D': 'days',
                    'M': 'months',
                    'Y': 'years'}

_units = (None, 's', 'm', 'h', 'D', 'M', 'Y')

_builtin_names = dir(builtins)


def get_value(v: Any) -> Union[str, int, float]:
    """
    If possible, get the int or float value of the passed object.
    :param v: an object for which to try to get the value.
    :return: the object's value (int or float) or the object itself.
    """
    v = str(v)

    if v in _builtin_names:
        return v

    try:
        return eval(v)

    except (NameError, SyntaxError):
        return v


def isCollection(obj: Any) -> bool:
    """
    Whether an object is a collection.
    :param obj: an object to test.
    :return: whether an object is a collection.
    """
    return True if hasattr(obj, '__iter__') and not issubclass(type(obj), str) else False


class Unit:
    """
    Simple class for storing a time point's unit.
    """
    _units_order = {'s': 1,
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

        self.value = value if value is not None else 's'

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

    def __eq__(self, other: 'Unit') -> bool:
        """
        Compare units with 'equal'.
        """
        return self.value == other.value


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
            self.value: 'NameUtils.DType' = time_point.value
            self.unit: Unit = time_point.unit

        else:
            self.value, self.unit = self.__parse(time_point)

    @staticmethod
    def __parse(time_point: Union[str, 'NameUtils.DType']) -> Tuple['NameUtils.DType', Unit]:
        """
        Get time point's value and unit.

        :param time_point: a time point's value given by the user.
        :return: tuple of value and unit.
        """
        _type_time_point = type(time_point)

        if _type_time_point in (int, float, np.int_, np.float_):
            return float(time_point), Unit(None)

        elif _type_time_point in (str, np.str_):
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

    def __gt__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'greater than'.
        """
        return self.unit > other.unit or (self.unit == other.unit and self.value > other.value)

    def __lt__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'lesser than'.
        """
        return self.unit < other.unit or (self.unit == other.unit and self.value < other.value)

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

    def __eq__(self, other: 'TimePoint') -> bool:
        """
        Compare units with 'equal'.
        """
        return self.unit == other.unit and self.value == other.value

    def __hash__(self) -> int:
        return hash(repr(self))


# Formatting & Conversion ------------------------------------------------
def to_list(value: Any) -> List[Any]:
    """
    Convert any object to a list.
    :param value: an object to convert to a list.
    :return: a list.
    """
    if isCollection(value):
        return list(value)

    else:
        return [value]


def to_tp_list(item: Any, reference_time_points: Optional[Collection[TimePoint]] = None) -> 'NameUtils.TimePointList':
    """
    Converts a given object to a tuple of TimePoints (or tuple of tuple of TimePoints ...).
    :param item: an object to convert to tuple of TimePoints.
    :param reference_time_points: an optional list of TimePoints that can exist. Used to parse the '*' character.
    :return: a (nested) tuple of TimePoints.
    """
    new_tp_list: 'NameUtils.TimePointList' = []

    if reference_time_points is None:
        reference_time_points = [TimePoint('0')]

    for v in to_list(item):
        if isCollection(v):
            new_tp_list.append(to_tp_list(v, reference_time_points))

        elif not isinstance(v, TimePoint) and v == '*':
            new_tp_list.append(tuple(reference_time_points))

        else:
            new_tp_list.append(TimePoint(v))

    return new_tp_list


def slicer_to_array(slicer: 'NameUtils.PreSlicer', reference_index: Collection, on_time_point: bool = False) -> \
        np.ndarray:
    """
    Format a slicer into an array of allowed values given in the 'reference_index' parameter.
    :param slicer: a PreSlicer object to format.
    :param reference_index: a collection of allowed values for the slicer.
    :param on_time_point: slicing on time points ?
    :return: an array of allowed values in the slicer.
    """
    if not isinstance(slicer, (slice, type(Ellipsis))):
        if isinstance(slicer, np.ndarray) and slicer.dtype == np.bool:
            # boolean array : extract values from reference_index
            return np.array(reference_index)[np.where(slicer)]

        elif not isCollection(slicer):
            # single value : convert to array (void array if value not in reference_index)
            slicer = TimePoint(slicer) if on_time_point else slicer
            return np.array([slicer]) if smart_isin(slicer, reference_index) else np.array([])

        else:
            # array of values : store values that are in reference_index
            slicer = to_tp_list(slicer) if on_time_point else slicer
            return np.array(slicer)[np.where(smart_isin(slicer, reference_index))]

    elif slicer == slice(None, None, None) or isinstance(slicer, type(Ellipsis)):
        # slice from start to end : take all values in reference_index
        return np.array(reference_index)

    elif isinstance(slicer, (slice, range)):
        # slice from specific start to end : get list of sliced values
        if on_time_point:
            slicer.start = TimePoint(slicer.start)
            slicer.stop = TimePoint(slicer.stop)
        return np.array(slice_or_range_to_list(slicer, reference_index))

    else:
        raise VTypeError(f"Invalid type {type(slicer)} for function 'slicer_to_array()'.")

def unique_in_list(c: Collection) -> Set:
    """
    Get the set of unique elements in a (nested) collection of elements.

    :param c: the (nested) collection of elements.
    :return: the set of unique elements in the (nested) collection of elements.
    """
    unique_values: Set = set()

    for value in c:
        if isCollection(value):
            unique_values = unique_values.union(unique_in_list(value))

        else:
            unique_values.add(value)

    return unique_values


def reformat_index(index: Union['NameUtils.PreSlicer',
                                Tuple['NameUtils.PreSlicer'],
                                Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer'],
                                Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer', 'NameUtils.PreSlicer']],
                   time_points_reference: Collection[TimePoint],
                   obs_reference: Collection,
                   var_reference: Collection) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format a sub-setting index into 3 arrays of selected (and allowed) values for time points, observations and
    variables. The reference collections are used to transform a PreSlicer into an array of selected values.
    :param index: an index to format.
    :param time_points_reference: a collection of allowed values for the time points.
    :param obs_reference: a collection of allowed values for the observations.
    :param var_reference: a collection of allowed values for the variables.
    :return: 3 arrays of selected (and allowed) values for time points, observations and variables.
    """
    if not isinstance(index, tuple):
        return slicer_to_array(index, time_points_reference, on_time_point=True), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 1:
        return slicer_to_array(index[0], time_points_reference, on_time_point=True), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 2:
        return slicer_to_array(index[0], time_points_reference, on_time_point=True), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(..., var_reference)

    else:
        return slicer_to_array(index[0], time_points_reference, on_time_point=True), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(index[2], var_reference)


def slice_or_range_to_list(s: Union[slice, range], _c: Collection[Any]) -> List[Any]:
    """
    Converts a slice or a range to a list of elements within that slice.
    :param s: a slice or range to convert.
    :param _c: a collection of elements to slice.
    """
    c = np.array(_c)
    if c.ndim != 1:
        raise ShapeError(f"The collection is {c.ndim}D, should be a 1D array.")

    sliced_list = []
    found_start = False
    current_step = 1

    start = s.start
    end = s.stop

    # get step value
    if s.step is None:
        step = 1

    else:
        step = s.step
        if not isinstance(step, int):
            raise VValueError(f"The 'step' value is {step}, should be an int.")

        if step == 0:
            raise VValueError("The 'step' value cannot be 0.")

    if step < 0:
        c = np.flip(c)

    # scan the collection of elements to extract values in the slice/range
    for element in c:
        if not found_start:
            # scan collection until the start element is found
            if element == start:
                sliced_list.append(element)
                found_start = True

        else:
            if element == end:
                break

            elif current_step == step:
                current_step = 1
                sliced_list.append(element)

            else:
                current_step += 1

    return sliced_list


# Representation ---------------------------------------------------------
def repr_array(arr: Union['NameUtils.DType', Sequence, range, slice, 'ellipsis']) -> str:
    """
    Get a short string representation of an array.
    :param: an array to represent.
    :return: a short string representation of the array.
    """
    if isinstance(arr, slice) or arr is ... or not isCollection(arr):
        return str(arr)

    else:
        arr = cast(Sequence, arr)
        if isinstance(arr, range) or len(arr) <= 4:
            return f"{str(arr)} ({len(arr)} value{'' if len(arr) == 1 else 's'} long)"

        elif isinstance(arr, pd.Series):
            return f"[{arr[0]} {arr[1]} ... {arr.iloc[-2]} {arr.iloc[-1]}] ({len(arr)} values long)"

        else:
            return f"[{arr[0]} {arr[1]} ... {arr[-2]} {arr[-1]}] ({len(arr)} values long)"


def repr_index(index: Union['NameUtils.PreSlicer', Tuple['NameUtils.PreSlicer'],
                            Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer'],
                            Tuple['NameUtils.PreSlicer', 'NameUtils.PreSlicer', 'NameUtils.PreSlicer'],
                            Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
        -> str:
    """
    Get a short string representation of a sub-setting index.
    :param index: a sub-setting index to represent.
    :return: a short string representation of the sub-setting index.
    """
    if isinstance(index, tuple):
        repr_string = f"Index of {len(index)} element{'' if len(index) == 1 else 's'} : "

        for element in index:
            repr_string += f"\n  \u2022 {repr_array(element) if isCollection(element) else element}"

        return repr_string

    else:
        return f"Index of 1 element : \n" \
               f"  \u2022 {repr_array(index) if isCollection(index) else index}"


# Identification ---------------------------------------------------------
def smart_isin(element: Any, target_collection: Collection) -> Union[bool, np.ndarray]:
    """
    Returns a boolean array of the same length as 'element' that is True where an element of 'element' is in
    'target_collection' and False otherwise.
    Here, elements are parsed to ints and floats when possible to assess equality. This allows to recognize that '0'
    is the same as 0.0 .
    :param element: 1D array of elements to test.
    :param target_collection: 1D array of allowed elements.
    :return: boolean array that is True where element is in target_collection.
    """
    if not isCollection(target_collection):
        raise VTypeError(f"Invalid type {type(target_collection)} for 'target_collection' parameter.")

    target_collection = set(e for e in target_collection)

    if not isCollection(element):
        element = [element]

    result = np.array([False for _ in range(len(element))])

    for i, e in enumerate(element):
        if len({e} & target_collection):
            result[i] = True

    return result if len(result) > 1 else result[0]


def match_time_points(tp_list: Collection, tp_index: Collection[TimePoint]) -> np.ndarray:
    """
    Find where in the tp_list the values in tp_index are present. This function parses the tp_list to understand the
        '*' character (meaning the all values in tp_index match) and tuples of time points.
    :param tp_list: the time points columns in a TemporalDataFrame.
    :param tp_index: a collection of target time points to match in tp_list.
    :return: a list of booleans of the same length as tp_list, where True indicates that a value in tp_list matched
        a value in tp_index.
    """
    tp_index = list(unique_in_list(tp_index))
    tp_list = to_tp_list(tp_list)

    mask = np.array([False for _ in range(len(tp_list))], dtype=bool)

    if len(tp_index):
        for tp_i, tp_value in enumerate(tp_list):
            if not isCollection(tp_value) and tp_value.value == '*':
                mask[tp_i] = True

            else:
                if isCollection(tp_value):
                    for one_tp_value in cast(Sequence, tp_value):
                        if smart_isin(one_tp_value, tp_index):
                            mask[tp_i] = True
                            break

                elif smart_isin(tp_value, tp_index):
                    mask[tp_i] = True

    return mask


def trim_time_points(tp_list: Collection, tp_index: Collection[TimePoint]) -> Tuple[Collection, Set]:
    """
    Remove from tp_list all time points not present in the reference tp_index. This function also works on collections
    of time points in tp_list.

    :param tp_list: a collection of time points or of collections of time points.
    :param tp_index: a reference collection of valid ime points.

    :return: tp_list without invalid time points.
    """
    tp_index = list(unique_in_list(tp_index))
    tp_list = to_tp_list(tp_list)

    result: List = [None for _ in range(len(tp_list))]
    excluded_elements: Set[TimePoint] = set()
    counter = 0

    for element in tp_list:
        if isCollection(element):
            res, excluded = trim_time_points(cast(Sequence, element), tp_index)
            excluded_elements = excluded_elements.union(excluded)

            if len(res):
                result[counter] = res
                counter += 1

        else:
            if element in tp_index:
                result[counter] = element
                counter += 1

            else:
                excluded_elements.add(element)

    return result[:counter], excluded_elements
