# coding: utf-8
# Created on 05/03/2021 16:43
# Author : matteo

# ====================================================
# imports
import numpy as np
from typing import Any, List, Optional, Collection, Union, Sequence, cast, Set, Tuple

from .NameUtils import TimePointList, PreSlicer
from vdata.utils import isCollection
from vdata.TimePoint import TimePoint
from .._IO import VTypeError, ShapeError, VValueError


# ====================================================
# code
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


def to_tp_list(item: Any, reference_time_points: Optional[Collection[TimePoint]] = None) -> 'TimePointList':
    """
    Converts a given object to a tuple of TimePoints (or tuple of tuple of TimePoints ...).
    :param item: an object to convert to tuple of TimePoints.
    :param reference_time_points: an optional list of TimePoints that can exist. Used to parse the '*' character.
    :return: a (nested) tuple of TimePoints.
    """
    new_tp_list: 'TimePointList' = []

    if reference_time_points is None or not len(reference_time_points):
        reference_time_points = [TimePoint('0')]

    for v in to_list(item):
        if isCollection(v):
            new_tp_list.append(to_tp_list(v, reference_time_points))

        elif not isinstance(v, TimePoint) and v == '*':
            if len(reference_time_points):
                if len(reference_time_points) == 1:
                    new_tp_list.append(reference_time_points[0])

                else:
                    new_tp_list.append(reference_time_points)

        else:
            new_tp_list.append(TimePoint(v))

    return new_tp_list


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


def slicer_to_array(slicer: 'PreSlicer', reference_index: Collection, on_time_point: bool = False) -> \
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


def reformat_index(index: Union['PreSlicer',
                                Tuple['PreSlicer'],
                                Tuple['PreSlicer', 'PreSlicer'],
                                Tuple['PreSlicer', 'PreSlicer', 'PreSlicer']],
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


# Identification ---------------------------------------------------------
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
