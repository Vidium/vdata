# coding: utf-8
# Created on 05/03/2021 16:44
# Author : matteo

# ====================================================
# imports
from typing import Collection, Set, List, Tuple, Sequence, cast

from ..utils import unique_in_list, to_tp_list
from vdata.utils import isCollection
from vdata.time_point import TimePoint


# ====================================================
# code
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
