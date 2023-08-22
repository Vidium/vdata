from types import EllipsisType
from typing import Collection, SupportsIndex, TypedDict, TypeVar, Union

import ch5mpy as ch
import numpy as np
import numpy.typing as npt

import vdata.timepoint as tp
from vdata.array_view import NDArrayView
from vdata.timedict import TimeDict

_T = TypeVar("_T")
_T_NP = TypeVar("_T_NP", bound=np.generic)

IF = Union[int, float, np.int_, np.float_]
IFS = Union[np.int_, int, np.float_, float, np.str_, str]

AnyNDArrayLike = Union[npt.NDArray[_T_NP], ch.H5Array[_T_NP], NDArrayView[_T_NP]]

AnyNDArrayLike_IF = Union[
    npt.NDArray[np.int_ | np.float_], ch.H5Array[np.int_ | np.float_], NDArrayView[np.int_ | np.float_]
]

NDArray_IFS = npt.NDArray[np.int_ | np.float_ | np.str_]
NDArrayLike_IFS = Union[npt.NDArray[np.int_ | np.float_ | np.str_], ch.H5Array[np.int_ | np.float_ | np.str_]]
AnyNDArrayLike_IFS = Union[
    npt.NDArray[np.int_ | np.float_ | np.str_],
    ch.H5Array[np.int_ | np.float_ | np.str_],
    NDArrayView[np.int_ | np.float_ | np.str_],
]

Collection_IFS = Collection[np.int_ | int | np.float_ | float | np.str_ | str]
DictLike = Union[dict[str, _T], ch.H5Dict[_T]]
AnyDictLike = Union[dict[str, _T], ch.H5Dict[_T], TimeDict]

Slicer = Union[IFS, tp.TimePoint, Collection[Union[IFS, tp.TimePoint]], range, slice, EllipsisType]
PreSlicer = Union[IFS, tp.TimePoint, Collection[Union[IFS, bool, tp.TimePoint]], range, slice, EllipsisType]
Indexer = Union[SupportsIndex, slice, npt.NDArray[np.int_], npt.NDArray[np.bool_] | None]


class AttrDict(TypedDict):
    name: str
    timepoints_column_name: str | None
    locked_indices: bool
    locked_columns: bool
    repeating_index: bool
