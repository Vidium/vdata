from types import EllipsisType
from typing import Sequence, Union

import numpy as np

from vdata.timepoint import TimePoint

IF = Union[int, float, np.int_, np.float_]
IFS_NP = Union[np.int_, np.float_, np.str_]
IFS = Union[np.int_, int, np.float_, float, np.str_, str]

Slicer = Union[Sequence[Union[int, float, str, bool, TimePoint]], range, slice]
PreSlicer = Union[int, float, str, TimePoint, Slicer, EllipsisType]

NumberType = Union[int, float, np.int_, np.float_]
