from __future__ import annotations

from types import EllipsisType
from typing import Any, Generic, Iterator, SupportsIndex, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from ch5mpy.indexing import Selection
from numpy._typing import _ArrayLikeInt_co, _ArrayLikeObject_co

_T = TypeVar('_T', bound=np.generic)
_NP_INDEX = Union[None, slice, EllipsisType, SupportsIndex, _ArrayLikeInt_co, 
                  tuple[None | slice | EllipsisType | _ArrayLikeInt_co | SupportsIndex, ...]]


class NDArrayView(Generic[_T]):
    """View on a numpy ndarray."""
    
    __slots__ = '_container', '_accession', '_index'
    
    # region magic methods
    def __init__(self,
                 container: Any,
                 accession: str,
                 index: _NP_INDEX | Selection) -> None:
        self._container = container
        self._accession = accession
        self._index = index if isinstance(index, Selection) else Selection.from_selector(
            index, getattr(container, accession).shape
        )

    def __repr__(self) -> str:
        return repr(self._view()) + '*'
        
    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[_T]:        
        if dtype is None:
            return self._view()
        
        return self._view().astype(dtype)
            
    def __getitem__(self, index: _NP_INDEX) -> NDArrayView[_T] | _T:
        sel = Selection.from_selector(index, self._array.shape).cast_on(self._index)
        
        if sel.size(self._array.shape):
            return NDArrayView(self._container, self._accession, sel)

        return cast(_T, self._array[sel.get()])
            
    def __len__(self) -> int:
        return len(self._view())
    
    def __contains__(self, key: Any) -> bool:
        return key in self._view()
    
    def __iter__(self) -> Iterator[_T]:
        return iter(self._view())
    
    def __eq__(self, __value: object) -> npt.NDArray[np.bool_]:
        return self._view().__eq__(__value)
    
    def __lt__(self, __value: _ArrayLikeObject_co) -> npt.NDArray[np.bool_]:
        return self._view().__lt__(__value)
    
    def __le__(self, __value: _ArrayLikeObject_co) -> npt.NDArray[np.bool_]:
        return self._view().__le__(__value)
    
    def __gt__(self, __value: _ArrayLikeObject_co) -> npt.NDArray[np.bool_]:
        return self._view().__gt__(__value)
    
    def __ge__(self, __value: _ArrayLikeObject_co) -> npt.NDArray[np.bool_]:
        return self._view().__ge__(__value)
        
    def __add__(self, other: Any) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._view() + other)
    
    def __sub__(self, other: Any) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._view() - other)

    def __mul__(self, other: Any) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._view() * other)

    def __truediv__(self, other: Any) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._view() / other)
        
    def __pow__(self, other: Any) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._view() ** other)
    
    # endregion
    
    # region predicates
    @property
    def size(self) -> int:
        return self._view().size
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._view().shape
    
    @property
    def dtype(self) -> np.dtype[_T]:
        return self._view().dtype
    
    @property
    def _array(self) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], getattr(self._container, self._accession))
    
    # endregion
    
    # region methods
    def _view(self) -> npt.NDArray[_T]:
        return cast(npt.NDArray[_T], self._array[self._index.get()])
    
    def copy(self) -> npt.NDArray[_T]:
        return self._view().copy()
        
    def astype(self, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        return self._view().astype(dtype)
    
    def min(self, 
            axis: int | tuple[int, ...] | None = None,
            out: npt.NDArray[Any] | None = None) -> _T | npt.NDArray[_T]:
        return self._view().min(axis=axis, out=out)
    
    def max(self, 
            axis: int | tuple[int, ...] | None = None,
            out: npt.NDArray[Any] | None = None) -> _T | npt.NDArray[_T]:
        return self._view().max(axis=axis, out=out)
    
    def mean(self, 
             axis: int | tuple[int, ...] | None = None,
             dtype: npt.DTypeLike | None = None,
             out: npt.NDArray[Any] | None = None) -> _T | npt.NDArray[_T]:
        return self._view().mean(axis=axis, dtype=dtype, out=out)         # type: ignore[arg-type, misc, no-any-return]
       
    def flatten(self) -> npt.NDArray[_T]:
        return self._view().flatten()
       
    # endregion
