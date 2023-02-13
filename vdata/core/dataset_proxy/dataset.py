# coding: utf-8
# Created on 25/10/2022 08:39
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from functools import wraps
from collections.abc import Sized
from vdata.h5pickle import Dataset
from vdata.h5pickle import File
from h5py import string_dtype

from typing import Any, Generic, Type

from vdata.time_point import TimePoint
from vdata.core.dataset_proxy.base import BaseDatasetProxy, SELECTOR, _VT, DATASET_DTYPE
from vdata.core.dataset_proxy.utils import auto_DatasetProxy


# ====================================================
# code
CAST = {int: np.int64,
        float: np.float64,
        str: np.str_}


def cast_int_to_float(func):
    """If needed, cast int dataset to float."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        if np.issubdtype(self._proxy.dtype, int) and not np.issubdtype(self._proxy.dtype, float):
            self.astype(float)

        return func(*args, **kwargs)

    return wrapper


class DatasetProxy(Sized, Generic[_VT]):
    """Proxy for h5py.Dataset objects."""

    # region magic methods
    def __init__(self,
                 data: Dataset | BaseDatasetProxy | DatasetProxy,
                 view_on: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
                 dtype: DATASET_DTYPE | None = None):
        if isinstance(data, BaseDatasetProxy):
            data, dtype = data.data, data.dtype

        elif isinstance(data, DatasetProxy):
            data, dtype = data.proxy.data, data.proxy.dtype

        self._proxy = auto_DatasetProxy(data, view_on, dtype)

    def __repr__(self) -> str:
        elem = 'element' if self._proxy.size == 1 else 'elements'
        return f"DatasetProxy([{' x '.join([str(e) for e in self._proxy.shape])} {elem}], dtype={self.dtype})"

    def __dir__(self) -> list[str]:
        return dir(self._proxy)

    def __getattr__(self,
                    item: str) -> Any:
        return getattr(object.__getattribute__(self, '_proxy'), item)

    def __getitem__(self,
                    item: SELECTOR) -> np.ndarray | _VT:
        return self._proxy[item]

    def __getstate__(self) -> dict[str, Any]:
        return {'view_on': self._proxy._view_on, 'dtype': self.dtype} | self.data.__getstate__()

    def __setstate__(self,
                     state: dict[str, Any]):
        self._proxy = auto_DatasetProxy(state['file'][state['name']],
                                        view_on=state['view_on'],
                                        dtype=state['dtype'])

    def __setitem__(self,
                    item: SELECTOR | tuple[SELECTOR, SELECTOR],
                    value: np.ndarray | _VT) -> None:
        self._proxy[item] = value

    @cast_int_to_float
    def __iadd__(self,
                 value: _VT) -> DatasetProxy:
        self._proxy += value
        return self

    @cast_int_to_float
    def __isub__(self,
                 value: _VT) -> DatasetProxy:
        self._proxy -= value
        return self

    @cast_int_to_float
    def __imul__(self,
                 value: _VT) -> DatasetProxy:
        self._proxy *= value
        return self

    @cast_int_to_float
    def __itruediv__(self,
                     value: _VT) -> DatasetProxy:
        self._proxy /= value
        return self

    def __add__(self,
                other: object) -> np.ndarray:
        return self._proxy + other

    def __sub__(self,
                other: object) -> np.ndarray:
        return self._proxy - other

    def __mul__(self,
                other: object) -> np.ndarray:
        return self._proxy * other

    def __truediv__(self,
                    other: object) -> np.ndarray:
        return self._proxy / other

    def __eq__(self,
               other: object) -> bool:
        if isinstance(other, DatasetProxy):
            return self._proxy == other.proxy

        return self._proxy == other

    def __len__(self) -> int:
        return len(self._proxy)

    # endregion

    # region attributes
    @property
    def proxy(self) -> BaseDatasetProxy:
        return self._proxy

    @property
    def dtype(self) -> DATASET_DTYPE:
        return self._proxy.dtype

    # endregion

    # region methods
    def close(self) -> None:
        self._proxy.close()

    def astype(self,
               dtype: DATASET_DTYPE | Type[int] | Type[float] | Type[str],
               replacement_data: np.ndarray | None = None) -> None:
        """
        In place data type conversion.
        """
        if dtype in CAST:
            # noinspection PyTypeChecker
            dtype = CAST[dtype]

        if self.dtype == dtype:
            return

        if not np.issubdtype(dtype, np.generic) or dtype == TimePoint:
            raise TypeError(f"Data type '{dtype}' is not supported.")

        h5_file: File = self._proxy.data.parent
        name = self._proxy.data.name
        shape_data = h5_file[name].shape

        if np.issubdtype(self._proxy.dtype, np.number):
            if np.issubdtype(dtype, np.str_):
                # replace num dataset with str dataset
                str_data = replacement_data if replacement_data is not None else h5_file[name][:]
                str_data = str_data.astype(str).astype('O')

                del h5_file[name]
                h5_file.create_dataset(name, shape=shape_data, dtype=string_dtype(), data=str_data)

                # update the proxy
                self._proxy = auto_DatasetProxy(h5_file[name], dtype=np.str_)

            elif dtype == TimePoint:
                raise NotImplementedError

            else:
                # replace num dataset with str dataset
                new_data = replacement_data if replacement_data is not None else h5_file[name][:]
                new_data = new_data.astype(dtype)

                del h5_file[name]
                h5_file.create_dataset(name, shape=shape_data, data=new_data)

                # update the proxy
                self._proxy = auto_DatasetProxy(h5_file[name], dtype=dtype)

        elif np.issubdtype(self._proxy.dtype, np.str_):
            if np.issubdtype(dtype, np.number):
                # replace str dataset with num dataset
                num_data = replacement_data if replacement_data is not None else h5_file[name][:]
                num_data = num_data.astype(dtype)

                del h5_file[name]
                h5_file.create_dataset(name, shape=shape_data, dtype=dtype, data=num_data)

                # update the proxy
                self._proxy = auto_DatasetProxy(h5_file[name], dtype=dtype)

            elif dtype == TimePoint:
                raise NotImplementedError

        else:
            raise TypeError(f"Type casting is not yet supported for type '{self._proxy.dtype}'.")

    # endregion
