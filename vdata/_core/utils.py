# coding: utf-8
# Created on 1/7/21 11:41 AM
# Author : matteo
# ====================================================
# imports
import numpy as np
from typing import Union, Tuple, Sequence, Collection, Any, List

from vdata.NameUtils import PreSlicer
from .._IO.errors import VTypeError, VValueError, ShapeError


# ====================================================
# code
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


def isCollection(obj: Any) -> bool:
    """
    Whether an object is a collection.
    :param obj: an object to test.
    :return: whether an object is a collection.
    """
    return True if hasattr(obj, '__iter__') and not issubclass(type(obj), str) else False


# Formatting & Conversion ------------------------------------------------
def reformat_index(index: Union[PreSlicer,
                                Tuple[PreSlicer],
                                Tuple[PreSlicer, PreSlicer],
                                Tuple[PreSlicer, PreSlicer, PreSlicer]],
                   time_points_reference: Collection,
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
        return slicer_to_array(index, time_points_reference), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 1:
        return slicer_to_array(index[0], time_points_reference), \
               slicer_to_array(..., obs_reference), \
               slicer_to_array(..., var_reference)

    elif isinstance(index, tuple) and len(index) == 2:
        return slicer_to_array(index[0], time_points_reference), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(..., var_reference)

    else:
        return slicer_to_array(index[0], time_points_reference), \
               slicer_to_array(index[1], obs_reference), \
               slicer_to_array(index[2], var_reference)


def slicer_to_array(slicer: PreSlicer, reference_index: Collection) -> np.ndarray:
    """
    Format a slicer into an array of allowed values given in the 'reference_index' parameter.
    :param slicer: a PreSlicer object to format.
    :param reference_index: a collection of allowed values for the slicer.
    :return: an array of allowed values in the slicer.
    """
    if not isinstance(slicer, (slice, type(Ellipsis))):
        if isinstance(slicer, np.ndarray) and slicer.dtype == np.bool:
            # boolean array : extract values from reference_index
            return np.array(reference_index)[np.where(slicer)]

        elif not isCollection(slicer):
            # single value : convert to array (void array if value not in reference_index)
            return np.array([slicer]) if smart_isin(slicer, reference_index) else np.array([])

        else:
            # array of values : store values that are in reference_index
            return np.array(slicer)[np.where(smart_isin(slicer, reference_index))]

    elif slicer == slice(None, None, None) or isinstance(slicer, type(Ellipsis)):
        # slice from start to end : take all values in reference_index
        return np.array(reference_index)

    elif isinstance(slicer, (slice, range)):
        # slice from specific start to end : get list of sliced values
        return np.array(slice_or_range_to_list(slicer, reference_index))

    else:
        raise VTypeError(f"Invalid type {type(slicer)} for function 'slicer_to_array()'.")


def slice_or_range_to_list(s: Union[slice, range], c: Collection[Any]) -> List[Any]:
    """
    Converts a slice or a range to a list of elements within that slice.
    :param s: a slice or range to convert.
    :param c: a collection of elements to slice.
    """
    c = np.array(c)
    if c.ndim != 1:
        raise ShapeError(f"The collection is {c.ndim}D, should be a 1D array.")

    sliced_list = []
    found_start = False
    current_step = 1

    start = get_value(s.start)
    end = get_value(s.stop)

    # get step value
    if s.step is None:
        step = 1

    else:
        step = get_value(s.step)
        if not isinstance(step, int):
            raise VValueError(f"The 'step' value is {step}, should be an int.")

        if step == 0:
            raise VValueError(f"The 'step' value cannot be 0.")

    if step < 0:
        c = np.flip(c)

    # scan the collection of elements to extract values in the slice/range
    for element in c:
        if not found_start:
            # scan collection until the start element is found
            if get_value(element) == start:
                sliced_list.append(element)
                found_start = True

        else:
            if get_value(element) == end:
                break

            elif current_step == step:
                current_step = 1
                sliced_list.append(element)

            else:
                current_step += 1

    return sliced_list


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


def to_str_list(item: Any) -> List:
    """
    Converts a given object to a list of string (or list of list of string ...).
    :param item: an object to convert to list of string.
    :return: a (nested) list of strings.
    """
    new_tp_list: List[Union[str, List]] = []
    for v in to_list(item):
        if isCollection(v):
            new_tp_list.append(to_str_list(v))

        else:
            new_tp_list.append(str(v))

    return new_tp_list


# Representation ---------------------------------------------------------
def repr_array(arr: Union[Sequence, range, slice]) -> str:
    """
    Get a short string representation of an array.
    :param: an array to represent.
    :return: a short string representation of the array.
    """
    if isinstance(arr, slice):
        return str(arr)

    elif isinstance(arr, range) or len(arr) <= 4:
        return f"{str(arr)} ({len(arr)} value{'' if len(arr) == 1 else 's'} long)"

    else:
        return f"[{arr[0]} {arr[1]} ... {arr[-2]} {arr[-1]}] ({len(arr)} values long)"


def repr_index(index: Union[PreSlicer, Tuple[PreSlicer], Tuple[PreSlicer, PreSlicer],
                            Tuple[PreSlicer, PreSlicer, PreSlicer], Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
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
def match_time_points(tp_list: Collection, tp_index: Collection[str]) -> np.ndarray:
    """
    Find where in the tp_list the values in tp_index are present. This function parses the tp_list to understand the
        '*' character (meaning the all values in tp_index match) and tuples of time points.
    :param tp_list: the time points columns in a TemporalDataFrame.
    :param tp_index: a collection of target time points to match in tp_list.
    :return: a list of booleans of the same length as tp_list, where True indicates that a value in tp_list matched
        a value in tp_index.
    """
    mask = np.array([False for _ in range(len(tp_list))], dtype=bool)

    tp_index = to_str_list(np.unique(tp_index))
    tp_list = to_str_list(tp_list)

    if len(tp_index):
        for tp_i, tp_value in enumerate(tp_list):
            if tp_value == '*':
                mask[tp_i] = True

            else:
                if isCollection(tp_value):
                    for one_tp_value in tp_value:
                        if smart_isin(one_tp_value, tp_index):
                            mask[tp_i] = True
                            break

                elif smart_isin(tp_value, tp_index):
                    mask[tp_i] = True

    return mask


def array_isin(array: np.ndarray, list_arrays: Union[np.ndarray, Collection[np.ndarray]]) -> bool:
    """
    Whether a given array is in a collection of arrays.
    :param array: an array.
    :param list_arrays: a collection of arrays.
    :return: whether the array is in the collection of arrays.
    """
    for target_array in list_arrays:
        if np.array_equal(array, target_array):
            return True

    return False


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

    target_collection = np.unique(target_collection)

    if not isCollection(element):
        element = [element]

    result = np.array([False for _ in range(len(element))])

    for i, e in enumerate(element):
        for e_t in target_collection:
            if get_value(e) == get_value(e_t):
                result[i] = True
                break

    return result if len(result) > 1 else result[0]




# TODO -------------------------------------------------------------------
# def reshape_to_3D(arr: ArrayLike_2D, time_points: Optional[Collection[str]], time_list: Optional[Collection[str]]) -> \
#         np.ndarray:
#     """
#     Reshape a 2D array-like object into a 3D array-like. Pandas DataFrames are first converted into numpy arrays.
#     By default, 2D arrays of shapes (n, m) are converted to 3D arrays of shapes (1, n, m). In this case, all the data
#         in the 2D array is considered to belong to the same time point.
#
#     Optionally, a collection of existing time points can be provided as the 'time_points' parameter. In this
#     collection, each time must be given only once. It will be used to split the 2D array into multiple arrays to
#     store in the final 3D array.
#     In addition to that, a collection of time points of the same length as the number of rows in the 2D array can be
#     given as the 'time_list' parameter to describe the 2D array. For each time point in 'time_points',
#     rows that match that time point in 'time_list' will be extracted from the 2D array to form a sub-array in the
#     final 3D array.
#     The 3D array will have shape (T, [n_1, ..., n_T], m) with :
#         * T, the number of unique time points in 'time_points'
#         * n_i, the number of rows in the sub-array for the time point i
#         * m, the number of columns
#
#     :param arr: a 2D array-like object.
#     :param time_points: a collection of existing and unique time points.
#     :param time_list: a collection of time points describing the rows of the 2D array
#     :return: a 3D numpy array.
#     """
#     # Check 2D array
#     if not isinstance(arr, (np.ndarray, pd.DataFrame)):
#         raise VTypeError(f"Type '{type(arr)}' is not allowed for conversion to 3D array.")
#
#     elif isinstance(arr, pd.DataFrame):
#         arr = np.array(arr)
#
#     # Check time points
#     if time_points is None:
#         time_points = ['0']
#
#         if time_list is not None:
#             generalLogger.warning("'time_list' parameter provided to reshape_to_3D() without specifying "
#                                   "'time_points', it will be ignored.")
#
#             time_list = np.zeros(len(arr))
#
#     else:
#         time_points = np.unique(time_points)
#
#         if time_list is None:
#             generalLogger.warning("'time_points' parameter provided to reshape_to_3D() without specifying "
#                                   "'time_list', all data is considered to come from the same time point.")
#
#     if time_list is None:
#         time_list = np.zeros(len(arr))
#
#     return np.array([arr[time_list == eval(str(TP))] for TP in time_points], dtype=object)
#
