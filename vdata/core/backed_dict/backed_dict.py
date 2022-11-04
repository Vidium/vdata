# coding: utf-8
# Created on 16/10/2022 12:05
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from vdata.h5pickle import File
from vdata.h5pickle import Group
from vdata.h5pickle import Dataset
from collections.abc import MutableMapping, KeysView, Iterable

from typing import Iterator, TypeVar, Any

from vdata.core.dataset_proxy import DatasetProxy
from vdata.utils import isCollection


# ====================================================
# code
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class BackedDictKeyIterator(Iterable[_KT]):
    """Class for iterating over keys in a BackedDict."""

    # region magic methods
    def __init__(self,
                 keys: KeysView[_KT]):
        self._keys = keys
        self._iterating = None

    def __iter__(self) -> Iterator[_KT]:
        self._iterating = iter(self._keys)
        return self._iterating

    def __next__(self) -> _KT:
        return next(self._iterating)

    # endregion


class BackedDict(MutableMapping[_KT, _VT]):
    """Class for managing dictionaries backed on h5 files."""

    # region magic methods
    def __init__(self,
                 file: File | Group):
        assert isinstance(file, (File, Group))
        assert file.attrs['type'] == 'dict'

        self._file = file

    def __repr__(self) -> str:
        if self.is_closed:
            return "Closed BackedDict{}"

        return f"BackedDict{set(self._file.keys())}"

    def __setitem__(self, key: _KT, value: _VT) -> None:
        # TODO : move writing functions to this package
        from vdata.read_write.write import write_Dict

        if key in self._file.keys():
            del self._file[key]

        if isinstance(value, dict):
            write_Dict(value, self._file, key=key)

        elif isCollection(value):
            value = np.array(value)
            if np.issubdtype(value.dtype, np.str_):
                value = value.astype('O')

            self._file[key] = value

        else:
            self._file[key] = value

    def __delitem__(self, key: _KT) -> None:
        del self._file[key]

    def __getitem__(self, key: _KT) -> _VT:
        from vdata.read_write.read import func_
        from vdata.read_write import H5GroupReader

        value = self._file[key]

        if isinstance(value, Group):
            type_ = value.attrs['type']

            # TODO : move reading function to this package
            if type_ == 'dict':
                return BackedDict(value)

            elif type_ in func_:
                # TODO : get rid of H5GroupReader
                return func_[type_](H5GroupReader(value), mode=value.file.mode)

            else:
                raise TypeError(f"Got unknown type '{value.attrs['type']}' when accessing key '{key}'.")

        elif isinstance(value, Dataset):
            if value.ndim >= 1:
                return DatasetProxy(value)

            else:
                value = value[()]
                if isinstance(value, bytes):
                    return value.decode()

                return value

        return value[()]

    def __len__(self) -> int:
        return len(self._file.keys())

    def __iter__(self) -> Iterator[_KT]:
        return iter(BackedDictKeyIterator(self._file.keys()))

    # endregion

    # region predicates
    @property
    def is_closed(self) -> bool:
        return not self._file.id.valid

    # endregion

    # region attributes
    @property
    def file(self) -> File | Group:
        return self._file

    # endregion

    # region methods
    def close(self) -> None:
        self._file.file.close()

    def copy(self) -> dict[str, Any]:
        """
        Build an in-memory copy of this BackedDict object.
        """
        def get_in_memory(value: Any) -> Any:
            if isinstance(value, BackedDict):
                return value.copy()

            elif isinstance(value, DatasetProxy):
                return value[:]

            return value

        return {k: get_in_memory(v) for k, v in self.items()}

    # endregion

