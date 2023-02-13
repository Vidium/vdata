# coding: utf-8
# Created on 22/10/2022 15:34
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy_indexed as npi
from vdata.h5pickle import Dataset
from numbers import Number
from itertools import chain
from abc import ABC, abstractmethod
from collections.abc import Sized

from typing import Iterable, TypeVar, Collection, Union, Any, Generic, Type

from vdata.time_point import TimePoint
from vdata.utils import isCollection

# ====================================================
# code
_VT = TypeVar('_VT')
_NumT = TypeVar('_NumT', Number, np.number)
_StrT = TypeVar('_StrT', bound=str)
_TimePointT = TypeVar('_TimePointT', bound=TimePoint)

SELECTOR = Union[int, Collection[int], np.ndarray, slice]
ACCESSOR = Union[int, np.ndarray, slice]

DATASET_DTYPE = Union[np.dtype, Type[TimePoint]]


# ==== base types =============================================================
class BaseDatasetProxy(Sized, Generic[_VT]):
    """Simple proxy for h5py.Dataset objects for performing inplace operations."""

    # region magic methods
    def __init__(self,
                 dataset: Dataset | BaseDatasetProxy,
                 view_on: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None):
        self._data = dataset.data if isinstance(dataset, BaseDatasetProxy) else dataset
        self._view_on = view_on

    def __repr__(self) -> str:
        view = '' if self._view_on is None else 'view of '
        return f"{self.__class__.__name__} on {view}'{self._data.file.filename}{self._data.name}' " \
               f"array of {self.shape} elements"

    def __dir__(self) -> Iterable[str]:
        return chain(dir(Dataset), dir(BaseDatasetProxy), ('_data', '_view_on'))

    def __getattr__(self,
                    item: str) -> Any:
        return getattr(self._data, item)

    @abstractmethod
    def __getitem__(self,
                    item: SELECTOR) -> np.ndarray | _VT:
        pass

    @abstractmethod
    def _getitem_core(self,
                      array: Dataset,
                      item: SELECTOR) -> np.ndarray | _VT:
        pass

    @abstractmethod
    def __setitem__(self,
                    item: SELECTOR | tuple[SELECTOR, SELECTOR],
                    value: np.ndarray | _VT) -> None:
        pass

    @abstractmethod
    def __iadd__(self,
                 value: _VT) -> BaseDatasetProxy:
        pass

    def __eq__(self,
               other: object) -> bool:
        if isinstance(other, BaseDatasetProxy):
            return self[:] == other[:]

        else:
            return self[:] == other

    def __len__(self) -> int:
        if self._view_on is not None:
            return self._view_on[0].shape[0]

        return self._data.shape[0]

    # endregion

    # region attributes
    @property
    def data(self) -> Dataset:
        return self._data

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    @abstractmethod
    def dtype(self) -> DATASET_DTYPE:
        pass

    # endregion

    # region methods
    def close(self) -> None:
        self._data.file.close()

    @abstractmethod
    def _parse_value(self,
                     value: np.ndarray | _VT) -> np.ndarray | _VT:
        pass

    @abstractmethod
    def _get(self,
             array: Dataset,
             index_0: ACCESSOR | None = None,
             index_1: ACCESSOR | None = None) -> np.ndarray | _VT:
        pass

    @abstractmethod
    def _set(self,
             array: Dataset,
             value: np.ndarray | _VT,
             index_0: ACCESSOR | None = None,
             index_1: ACCESSOR | None = None) -> None:
        pass

    # endregion


class _NumDatasetProxy(ABC, BaseDatasetProxy, Generic[_NumT]):
    """Simple proxy for numerical h5py.Dataset objects for performing inplace operations."""

    # region magic methods
    def __getitem__(self, item) -> np.ndarray | _NumT:
        return self._getitem_core(self._data, item)

    def __iadd__(self,
                 value: _NumT) -> _NumDatasetProxy:
        new_values = self._get(self._data) + value
        self._set(self._data, new_values)
        return self

    def __add__(self,
                value: object) -> np.ndarray | _NumT:
        return self._get(self._data) + value

    def __isub__(self,
                 value: _VT) -> _NumDatasetProxy:
        new_values = self._get(self._data) - value
        self._set(self._data, new_values)
        return self

    def __sub__(self,
                value: object) -> np.ndarray | _NumT:
        return self._get(self._data) - value

    def __imul__(self,
                 value: _VT) -> _NumDatasetProxy:
        new_values = self._get(self._data) * value
        self._set(self._data, new_values)
        return self

    def __mul__(self,
                value: object) -> np.ndarray | _NumT:
        return self._get(self._data) * value

    def __itruediv__(self,
                     value: _VT) -> _NumDatasetProxy:
        new_values = self._get(self._data) / value
        self._set(self._data, new_values)
        return self

    def __truediv__(self,
                    value: object) -> np.ndarray | _NumT:
        return self._get(self._data) / value

    # endregion

    # region attributes
    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    # endregion

    # region methods
    def _parse_value(self,
                     value: np.ndarray | _VT) -> np.ndarray | _VT:
        if isinstance(value, np.ndarray):
            return value.astype(self._data.dtype)

        return value

    # endregion


class _StrDatasetProxy(ABC, BaseDatasetProxy, Generic[_StrT]):
    """Simple proxy for string h5py.Dataset objects for performing inplace operations."""

    # region magic methods
    def __init__(self,
                 dataset: Dataset | _StrDatasetProxy,
                 view_on: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__(dataset, view_on)

        self._data_str = self._data.asstr(encoding='utf-8')

    def __getitem__(self, item) -> np.ndarray | _StrT:
        subset = self._getitem_core(self._data_str, item)

        # cast as string if multiple values were selected
        if isinstance(subset, np.ndarray):
            return subset.astype(str)

        return str(subset)

    def __iadd__(self,
                 value: str) -> _StrDatasetProxy:
        new_values = self._get(self._data_str) + value
        self._set(self._data, new_values)
        return self

    def __array__(self, **kwargs) -> np.ndarray:
        return self._get(self._data_str)

    # endregion

    # region attributes
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Get array data type."""

    # endregion

    # region methods
    def _parse_value(self,
                     value: np.ndarray | _VT) -> np.ndarray | _VT:
        if isinstance(value, np.ndarray):
            return value.astype(str)

        return value

    def unique(self) -> np.ndarray:
        return np.unique(self._get(self._data_str))

    # endregion


class _TPDatasetProxy(ABC, BaseDatasetProxy, Generic[_TimePointT]):
    """Simple proxy for TimePoint objects for performing inplace operations."""

    # region magic methods
    def __getitem__(self, item) -> np.ndarray | _TimePointT:
        subset = self._getitem_core(self._data, item)

        if isinstance(subset, np.ndarray):
            return np.array([TimePoint(tp.decode()) for tp in subset])

        return TimePoint(subset.decode())

    def __iadd__(self, value: TimePoint) -> _TPDatasetProxy:
        raise NotImplemented

    # endregion

    # region attributes
    @property
    def dtype(self) -> Type[TimePoint]:
        return TimePoint

    # endregion

    # region methods
    def _parse_value(self,
                     value: np.ndarray | _VT) -> np.ndarray | _VT:
        if isinstance(value, np.ndarray):
            return value.astype(str)

        return value

    def unique(self) -> np.ndarray:
        return np.array(sorted([TimePoint(tp.decode()) for tp in np.unique(self._get(self._data))]))

    def is_equal(self,
                 timepoint: _TimePointT) -> np.ndarray:
        return np.equal(self._get(self._data), str(timepoint).encode())

    # endregion


# ==== dimension mixins =======================================================
class _Dataset1DMixin(ABC, BaseDatasetProxy):
    """Simple proxy for 1 dimensional h5py.Dataset objects for performing inplace operations."""

    # region magic methods
    def __init__(self,
                 dataset: Dataset | _Dataset1DMixin,
                 view_on: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__(dataset, view_on)

        assert self._data.ndim == 1, f"Can't instantiate 1D dataset proxy for a {self._data.ndim}D dataset."
        assert self._view_on is None or isinstance(self._view_on, np.ndarray), "Invalid view : should be a None " \
                                                                               "or a numpy array."

    @staticmethod
    def _valid_selector(selector: SELECTOR) -> ACCESSOR:
        if isCollection(selector):
            selector = np.array(selector)

        if isinstance(selector, np.ndarray) and selector.ndim != 1:
            raise IndexError

        return selector

    def _getitem_core(self,
                      array: Dataset,
                      item: SELECTOR) -> np.ndarray | _VT:
        return self._get(array, self._valid_selector(item))

    def __setitem__(self,
                    item: SELECTOR,
                    value: np.ndarray | _VT) -> None:
        self._set(self._data, value, self._valid_selector(item))

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int]:
        if self._view_on is not None:
            return len(self._view_on),

        return self._data.shape

    # endregion

    # region methods
    def _parse_index(self,
                     index: ACCESSOR | None) -> ACCESSOR:
        if index is None:
            index = slice(None)

        # if view on dataset, return the selected indices in the view
        if self._view_on is not None:
            return self._view_on[index]

        # otherwise, return a slice over the whole dataset if possible, or the selected indices in the dataset
        if isinstance(index, slice) and index == slice(None):
            return index

        return np.arange(len(self._data))[index]

    def _get(self,
             array: Dataset,
             index_0: ACCESSOR | None = None,
             index_1: None = None) -> np.ndarray | _VT:
        if index_1 is not None:
            raise IndexError

        index_0 = self._parse_index(index_0)

        # get whole array
        if isinstance(index_0, slice) and index_0 == slice(None):
            return array[:]

        # get value(s) in array
        return array[index_0]

    def _set(self,
             array: Dataset,
             value: np.ndarray | _VT,
             index_0: ACCESSOR | None = None,
             index_1: None = None) -> None:
        if index_1 is not None:
            raise IndexError

        index_0 = self._parse_index(index_0)
        value = self._parse_value(value)

        # set values for whole array
        if isinstance(index_0, slice) and index_0 == slice(None):
            array[:] = value

        # set values in array
        array[index_0] = value

    # endregion


class _Dataset2DMixin(ABC, BaseDatasetProxy):
    """Simple proxy for 1 dimensional h5py.Dataset objects for performing inplace operations."""

    # region magic methods
    def __init__(self,
                 dataset: Dataset | _Dataset2DMixin,
                 view_on: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__(dataset, view_on)

        assert self._data.ndim == 2, f"Can't instantiate 2D dataset proxy for a {self._data.ndim}D dataset."
        assert self._view_on is None or (isinstance(self._view_on, tuple) and len(self._view_on) == 2
                                         and isinstance(self._view_on[0], np.ndarray)
                                         and isinstance(self._view_on[1], np.ndarray)),\
            "Invalid view : should be a None or a 2-tuple of numpy arrays."

    @staticmethod
    def _valid_selectors(selector: SELECTOR | tuple[SELECTOR, SELECTOR]) -> tuple[ACCESSOR, ACCESSOR]:
        # tuple of selectors
        if isinstance(selector, tuple) and len(selector) == 2:
            sel0, sel1 = selector

        # one selector
        else:
            sel0, sel1 = selector, None

        if isCollection(sel0):
            sel0 = np.array(sel0)

        if isCollection(sel1):
            sel1 = np.array(sel1)

        # if item_0 in column vector form, reshape as row vector
        if isinstance(sel0, np.ndarray) and sel0.ndim == 2 and sel0.shape[1] == 1:
            sel0 = sel0.reshape(sel0.shape[0])

        return sel0, sel1

    @staticmethod
    def _order(accessor: ACCESSOR | None) -> ACCESSOR | None:
        if accessor is not None and isinstance(accessor, np.ndarray):
            return np.sort(accessor)

        return accessor

    @staticmethod
    def _reorder(accessor: ACCESSOR | None) -> ACCESSOR | None:
        if accessor is None:
            return slice(None)

        if isinstance(accessor, slice):
            if accessor.step is None or accessor.step > 0:
                step = None

            else:
                step = -1

            return slice(None, None, step)

        elif isinstance(accessor, np.ndarray):
            if accessor.size == 0:
                return accessor

            return npi.indices(sorted(accessor), accessor)

        return accessor

    def _getitem_core(self,
                      array: Dataset,
                      item: SELECTOR | tuple[SELECTOR, SELECTOR]) -> np.ndarray | _VT:
        return self._get(array, *self._valid_selectors(item))

    def __setitem__(self,
                    item: SELECTOR | tuple[SELECTOR, SELECTOR],
                    value: np.ndarray | _VT) -> None:
        self._set(self._data, value, *self._valid_selectors(item))

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, int]:
        if self._view_on is not None:
            return len(self._view_on[0]), len(self._view_on[1])

        return self._data.shape

    # endregion

    # region methods
    def _parse_index(self,
                     index: ACCESSOR | None,
                     axis: int) -> ACCESSOR:
        if index is None:
            index = slice(None)

        # if view on dataset, return the selected indices in the view
        if self._view_on is not None:
            return self._view_on[axis][index]

        # otherwise, return a slice over the whole dataset if possible, or the selected indices in the dataset
        if isinstance(index, slice) and index == slice(None):
            return index

        return np.arange(self._data.shape[axis])[index]

    def _get(self,
             array: Dataset,
             index_0: ACCESSOR | None = None,
             index_1: ACCESSOR | None = None) -> np.ndarray | _VT:
        index_0, index_1 = self._parse_index(index_0, 0), self._parse_index(index_1, 1)

        index_0_ordered, index_1_ordered = self._order(index_0), self._order(index_1)

        # get whole array
        if isinstance(index_0_ordered, slice) and index_0_ordered == slice(None) \
                and isinstance(index_1_ordered, slice) and index_1_ordered == slice(None):
            return array[:][self._reorder(index_0_ordered), self._reorder(index_1_ordered)]

        # get value(s) in array
        indexed_array = array[index_0_ordered]
        if indexed_array.ndim == 2:
            return indexed_array[self._reorder(index_0)][:, index_1_ordered][:, self._reorder(index_1)]

        # no need to reorder index_0 here: if array is 1D, index_0 was only ONE value
        indexed_array = indexed_array[index_1_ordered]
        if isinstance(indexed_array, np.ndarray):
            return indexed_array[self._reorder(index_1)]

        return indexed_array

    def _set(self,
             array: Dataset,
             value: np.ndarray | _VT,
             index_0: ACCESSOR | None = None,
             index_1: ACCESSOR | None = None) -> None:
        index_0, index_1 = self._parse_index(index_0, 0), self._parse_index(index_1, 1)
        value = self._parse_value(value)

        # set values for whole array
        if isinstance(index_0, slice) and index_0 == slice(None)\
                and isinstance(index_1, slice) and index_1 == slice(None):
            array[:] = value

        # set values for multiple columns (one at a time since smart indexing is not allowed)
        elif isinstance(index_1, np.ndarray):
            for value_index, array_index in enumerate(index_1):
                array[index_0, array_index] = value[:, value_index]

        # set values for a single column
        elif isinstance(index_1, slice) and index_1 == slice(None):
            array[index_0] = value

        else:
            array[index_0, index_1] = value

    # endregion
