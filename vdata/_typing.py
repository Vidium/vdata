from types import EllipsisType
from typing import Collection, TypedDict, TypeVar, Union

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

from vdata.timepoint import TimePoint

_T = TypeVar('_T')
_T_NP = TypeVar('_T_NP', bound=np.generic)

IF = Union[int, float, np.int_, np.float_]
IFS = Union[np.int_, int, np.float_, float, np.str_, str]

NDArray_IFS = Union[npt.NDArray[np.int_], npt.NDArray[np.float_], npt.NDArray[np.str_]]
H5Array_IFS = Union[ch.H5Array[np.int_], ch.H5Array[np.float_], ch.H5Array[np.str_]]
NDArrayLike = Union[npt.NDArray[_T_NP], ch.H5Array[_T_NP]]
NDArrayLike_IFS = Union[NDArray_IFS, H5Array_IFS]
Collection_IFS = Union[Collection[np.int_], Collection[int], 
                       Collection[np.float_], Collection[float],
                       Collection[np.str_], Collection[str]]
DictLike = Union[dict[str, _T], ch.H5Dict[_T]]

Slicer = Union[IFS, TimePoint, Collection[Union[IFS, TimePoint]], range, slice, EllipsisType]
PreSlicer = Union[IFS, TimePoint, Collection[Union[IFS, bool, TimePoint]], range, slice, EllipsisType]

NumberType = Union[int, float, np.int_, np.float_]


class AttrDict(TypedDict):
    name: str
    timepoints_column_name: str | None
    locked_indices: bool
    locked_columns: bool
    repeating_index: bool
