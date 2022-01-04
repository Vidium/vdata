# coding: utf-8
# Created on 05/03/2021 16:44
# Author : matteo

# ====================================================
# imports
from typing import Collection, cast, Optional, Union

from ..utils import unique_in_list, to_tp_list
from vdata.utils import isCollection
from vdata.time_point import TimePoint


# ====================================================
# code
TP_list = list[Optional[Union[TimePoint, list['TP_list']]]]


def trim_time_points(tp_list: Union[TimePoint, Collection[TimePoint]],
                     tp_index: Collection[TimePoint]) -> tuple[TP_list, set[TimePoint]]:
    """
    Remove from tp_list all time-points not present in the reference tp_index. This function also works on collections
    of time points in tp_list.

    Args:
        tp_list: a collection of time points or of collections of time points.
        tp_index: a reference collection of valid ime points.

    Returns:
        tp_list without invalid time points.
    """
    tp_index = list(unique_in_list(tp_index))
    tp_list = to_tp_list(tp_list)

    result: TP_list = [None for _ in range(len(tp_list))]
    excluded_elements: set[TimePoint] = set()
    counter = 0

    for element in tp_list:
        if isCollection(element):
            res, excluded = trim_time_points(cast(Collection, element), tp_index)
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
