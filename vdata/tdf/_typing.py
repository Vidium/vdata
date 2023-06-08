from types import EllipsisType
from typing import Collection, TypedDict, TypeVar, Union

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

from vdata._typing import IFS
from vdata.timepoint import TimePoint

_T = TypeVar('_T', bound=np.generic)

NDArrayLike = Union[npt.NDArray[_T], ch.H5Array[_T]]
SLICER = Union[IFS, TimePoint, Collection[Union[IFS, TimePoint]], range, slice, EllipsisType]

class AttrDict(TypedDict):
    name: str
    timepoints_column_name: str | None
    locked_indices: bool
    locked_columns: bool
    repeating_index: bool
    