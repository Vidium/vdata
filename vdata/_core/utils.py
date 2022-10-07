# coding: utf-8
# Created on 05/03/2021 16:43
# Author : matteo

# ====================================================
# imports
import numpy as np
from numbers import Number

from typing import Any, Optional, Collection, Union, Sequence, cast

from .name_utils import TimePointList, PreSlicer
from vdata.utils import isCollection, repr_array
from vdata.time_point import TimePoint
from ..IO import VTypeError, ShapeError, VValueError


# ====================================================
# code
# Formatting & Conversion ------------------------------------------------
def to_list(value: Any) -> list[Any]:
    """
    Convert any object to a list.
    :param value: an object to convert to a list.
    :return: a list.
    """
    if isCollection(value):
        return list(value)

    else:
        return [value]


def list_to_tp_list_strict(item: Sequence[Any]) -> list[TimePoint]:
    """
    Convert a list of elements into a list of TimePoints.

    :param item: a list of elements to convert into TimePoints.

    :return: a list of TimePoints.
    """
    return [TimePoint(e) for e in item]


def to_tp_list(item: Any, reference_timepoints: Optional[Sequence[TimePoint]] = None) -> 'TimePointList':
    """
    Converts a given object to a tuple of TimePoints (or tuple of tuple of TimePoints ...).

    :param item: an object to convert to tuple of TimePoints.
    :param reference_timepoints: an optional list of TimePoints that can exist. Used to parse the '*' character.

    :return: a (nested) tuple of TimePoints.
    """
    new_tp_list: 'TimePointList' = []

    if reference_timepoints is None or not len(reference_timepoints):
        reference_timepoints = [TimePoint('0h')]

    for v in to_list(item):
        if isCollection(v):
            new_tp_list.append(list_to_tp_list_strict(v))

        elif isinstance(v, TimePoint):
            new_tp_list.append(v)

        elif v == '*':
            if len(reference_timepoints):
                if len(reference_timepoints) == 1:
                    new_tp_list.append(reference_timepoints[0])

                else:
                    new_tp_list.append(reference_timepoints)

        elif isinstance(v, (Number, np.number)):
            new_tp_list.append(TimePoint(str(v) + 'h'))

        else:
            new_tp_list.append(TimePoint(v))

    return new_tp_list


def slice_or_range_to_list(s: Union[slice, range], _c: Collection[Any]) -> list[Any]:
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


def slicer_to_array(slicer: 'PreSlicer',
                    reference_index: Collection,
                    on_time_point: bool = False) -> \
        Optional[np.ndarray]:
    """
    Format a slicer into an array of allowed values given in the 'reference_index' parameter.

    Args:
        slicer: a PreSlicer object to format.
        reference_index: a collection of allowed values for the slicer.
        on_time_point: slicing on time points ?

    Returns:
        An array of allowed values in the slicer.
    """
    if not isinstance(slicer, (slice, type(Ellipsis))):
        if isinstance(slicer, np.ndarray) and slicer.dtype == bool:
            # boolean array : extract values from reference_index
            return np.array(reference_index)[np.where(slicer.flatten())]

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
        # return np.array(reference_index)
        return None

    elif isinstance(slicer, (slice, range)):
        # slice from specific start to end : get list of sliced values
        if on_time_point:
            slicer.start = TimePoint(float(slicer.start))
            slicer.stop = TimePoint(float(slicer.stop))
        return np.array(slice_or_range_to_list(slicer, reference_index))

    raise VTypeError(f"Invalid type {type(slicer)} for function 'slicer_to_array()'.")


def reformat_index(index: Union['PreSlicer',
                                tuple['PreSlicer'],
                                tuple['PreSlicer', 'PreSlicer'],
                                tuple['PreSlicer', 'PreSlicer', 'PreSlicer']],
                   timepoints_reference: Optional[Collection[TimePoint]],
                   obs_reference: Optional[Collection],
                   var_reference: Optional[Collection]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format a sub-setting index into 3 arrays of selected (and allowed) values for time points, observations and
    variables. The reference collections are used to transform a PreSlicer into an array of selected values.
    :param index: an index to format.
    :param timepoints_reference: a collection of allowed values for the time points.
    :param obs_reference: a collection of allowed values for the observations.
    :param var_reference: a collection of allowed values for the variables.
    :return: 3 arrays of selected (and allowed) values for time points, observations and variables.
    """
    if not isinstance(index, tuple):
        return slicer_to_array(index, timepoints_reference, on_time_point=True), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 1:
        return slicer_to_array(index[0], timepoints_reference, on_time_point=True), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 2:
        return slicer_to_array(index[0], timepoints_reference, on_time_point=True), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(..., var_reference)

    else:
        return slicer_to_array(index[0], timepoints_reference, on_time_point=True), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(index[2], var_reference)


# Identification ---------------------------------------------------------
def unique_in_list(c: Collection) -> set:
    """
    Get the set of unique elements in a (nested) collection of elements.

    :param c: the (nested) collection of elements.
    :return: the set of unique elements in the (nested) collection of elements.
    """
    unique_values: set = set()

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


def match_timepoints(tp_list: Collection, tp_index: Collection[TimePoint]) -> np.ndarray:
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


# Representation ---------------------------------------------------------
def repr_index(index: Union['PreSlicer', tuple['PreSlicer'],
                            tuple['PreSlicer', 'PreSlicer'],
                            tuple['PreSlicer', 'PreSlicer', 'PreSlicer'],
                            tuple[np.ndarray, np.ndarray, np.ndarray]]) \
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
