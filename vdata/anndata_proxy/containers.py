from __future__ import annotations

import operator as op
from functools import partialmethod
from typing import Any, Callable, ItemsView, KeysView, MutableMapping, Union, ValuesView, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss
from h5dataframe import H5DataFrame

import vdata
from vdata._typing import AnyNDArrayLike, AnyNDArrayLike_IFS
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase

_NP_IF = Union[np.int_, np.float_]


class TemporalDataFrameContainerProxy:

    __slots__ = "_vdata", "_tdfs", "_name"

    # region magic methods
    def __init__(self, vdata: vdata.VData | vdata.VDataView, name: str) -> None:
        self._vdata = vdata
        self._tdfs: MutableMapping[str, TemporalDataFrameBase] = getattr(vdata, name)
        self._name = name.capitalize()

    def __repr__(self) -> str:
        return f"{self._name} with keys: {', '.join(self._tdfs.keys())}"

    def __getitem__(self, key: str) -> ArrayStack2DProxy:
        return ArrayStack2DProxy(self._tdfs[str(key)].values_num, self._tdfs[str(key)].values_str)

    def __setitem__(self, key: str, value: npt.NDArray[Any]) -> None:
        self._tdfs[key] = TemporalDataFrame(
            value,
            index=self._vdata.obs.index,
            columns=self._vdata.var.index,
            timepoints=self._vdata.obs.timepoints_column,
        )

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._tdfs.keys()

    def values(self) -> ValuesView[TemporalDataFrameBase]:
        return self._tdfs.values()

    def items(self) -> ItemsView[str, TemporalDataFrameBase]:
        return self._tdfs.items()

    # endregion


class ArrayStack2DProxy:

    __slots__ = "_array_numeric", "_array_string"
    ndim = 2

    # region magic methods
    def __init__(self, array_numeric: AnyNDArrayLike[_NP_IF], array_string: AnyNDArrayLike[np.str_]) -> None:
        self._array_numeric = array_numeric if array_numeric.size else None
        self._array_string = array_string if array_string.size else None

    def __repr__(self) -> str:
        return repr(self.stack(n=5)) + "\n..." if self.shape[1] > 5 else ""

    def __getitem__(self, item: Any) -> Any:
        if self._array_numeric is not None and self._array_string is not None:
            return np.hstack((self._array_numeric.astype(object), self._array_string))[item]

        if self._array_numeric is None and self._array_string is not None:
            return self._array_string[item]

        if self._array_numeric is not None and self._array_string is None:
            return self._array_numeric[item]

        return np.empty((0, 0))[item]

    def _op(self, other: Any, operation: Callable[[Any, Any], Any]) -> npt.NDArray[_NP_IF]:
        if self._array_string is not None:
            raise TypeError(f"Cannot apply {operation.__name__} with string array.")

        if self._array_numeric is None:
            return np.empty((0, 0))

        return cast(npt.NDArray[_NP_IF], operation(self._array_numeric, other))

    __add__ = __radd__ = partialmethod(_op, operation=op.add)
    __sub__ = __rsub__ = partialmethod(_op, operation=op.sub)
    __mul__ = __rmul__ = partialmethod(_op, operation=op.mul)
    __truediv__ = __rtruediv__ = partialmethod(_op, operation=op.truediv)
    __gt__ = partialmethod(_op, operation=op.gt)
    __ge__ = partialmethod(_op, operation=op.ge)
    __lt__ = partialmethod(_op, operation=op.lt)
    __le__ = partialmethod(_op, operation=op.le)

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        if dtype is None:
            return self.stack()

        return self.stack().astype(dtype)

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, int]:
        if self._array_numeric is not None and self._array_string is not None:
            return (self._array_numeric.shape[0], self._array_numeric.shape[1] + self._array_string.shape[1])

        if self._array_numeric is None and self._array_string is not None:
            return self._array_string.shape  # type: ignore[return-value]

        if self._array_numeric is not None and self._array_string is None:
            return self._array_numeric.shape  # type: ignore[return-value]

        return (0, 0)

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._array_numeric is None and self._array_string is None:
            return np.dtype(np.float64)

        if self._array_numeric is None and self._array_string is not None:
            return self._array_string.dtype

        if self._array_numeric is not None and self._array_string is None:
            return self._array_numeric.dtype

        return np.dtype(np.object_)

    # endregion

    # region methods
    def stack(self, n: int | None = None) -> npt.NDArray[Any]:
        _subset = slice(None) if n is None else slice(0, n)

        if self._array_numeric is not None and self._array_string is not None:
            return np.hstack((self._array_numeric[_subset].astype(object), self._array_string[_subset]))

        if self._array_numeric is None and self._array_string is not None:
            return np.array(self._array_string[_subset])

        if self._array_numeric is not None and self._array_string is None:
            return np.array(self._array_numeric[_subset])

        return np.empty((0, 0))

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Any:
        if self._array_string is not None:
            raise TypeError("Cannot apply sum with string array.")

        if self._array_numeric is None:
            return np.empty((0, 0))

        return self._array_numeric.sum(axis=axis)

    # endregion


class H5DataFrameContainerProxy:

    __slots__ = "_h5dfs", "_index", "_columns", "_name", "_sparse_matrices"

    # region magic methods
    def __init__(
        self,
        h5dfs: MutableMapping[str, H5DataFrame],
        name: str,
        index: AnyNDArrayLike_IFS,
        columns: AnyNDArrayLike_IFS | None = None,
    ) -> None:
        self._h5dfs = h5dfs
        self._index = index
        self._columns = columns
        self._name = name

        # FIXME : find better way
        self._sparse_matrices: set[str] = set()

    def __repr__(self) -> str:
        return f"{self._name} with keys: {', '.join(self._h5dfs.keys())}"

    def __getitem__(self, key: str) -> npt.NDArray[Any] | ss.csr_matrix:
        if str(key) in self._sparse_matrices:
            return ss.csr_matrix(self._h5dfs[str(key)].values)

        return self._h5dfs[str(key)].values

    def __setitem__(self, key: str, value: npt.NDArray[Any]) -> None:
        if ss.issparse(value):
            self._sparse_matrices.add(str(key))

        self._h5dfs[str(key)] = H5DataFrame(value, index=self._index, columns=self._columns)

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._h5dfs.keys()

    # endregion
