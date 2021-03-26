# coding: utf-8
# Created on 08/03/2021 15:03
# Author : matteo

# ====================================================
# imports
from typing import Union, Sequence, List

from ..TimePoint import TimePoint

# ====================================================
# code
Slicer = Union[Sequence[Union[int, float, str, bool, TimePoint]], range, slice]
PreSlicer = Union[int, float, str, TimePoint, Slicer, 'ellipsis']

TimePointList = List[Union[TimePoint, List[TimePoint]]]
