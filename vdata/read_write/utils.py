# coding: utf-8
# Created on 19/01/2021 15:36
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Union, AbstractSet, ValuesView, Any, Literal

from vdata.h5pickle import H5Group


# ====================================================
# code
class H5GroupReader:
    """
    Class for reading a h5py File, Group or Dataset
    """

    def __init__(self, group: 'H5Group'):
        """
        Args:
            group: a h5py File, Group or Dataset
        """
        self.group = group

    def __getitem__(self, key: Union[str, slice, 'ellipsis', tuple[()]]) \
            -> Union['H5GroupReader', np.ndarray, str, int, float, bool, type]:
        """
        Get a subgroup from the group, identified by a key

        Args:
            key: the name of the subgroup
        """
        if isinstance(key, slice):
            return self._check_type(self.group[:])
        elif key is ...:
            return self._check_type(self.group[...])
        elif key == ():
            return self._check_type(self.group[()])
        else:
            return H5GroupReader(self.group[key])

    def __enter__(self):
        self.group.__enter__()
        return self

    def __exit__(self, *_):
        self.group.__exit__()

    def __sizeof__(self) -> int:
        return H5GroupReader.__basicsize__ + sum([])

    def close(self) -> None:
        self.group.file.close()

    @property
    def name(self) -> str:
        """
        Get the name of the group.

        Returns:
            The group's name.
        """
        return self.group.name

    @property
    def filename(self) -> str:
        """
        Get the filename of the group.

        Returns:
            The group's filename.
        """
        return self.group.file.filename

    @property
    def mode(self) -> str:
        """
        Get the reading mode for the group.

        Returns:
            The reading mode for the group.
        """
        return self.group.file.mode

    @property
    def parent(self) -> 'H5GroupReader':
        """
        Get the parent H5GroupReader.

        Returns:
            The parent H5GroupReader.
        """
        return H5GroupReader(self.group.parent)

    def keys(self) -> AbstractSet:
        """
        Get keys of the group.

        Returns:
            The keys of the group.
        """
        return self.group.keys()

    def values(self) -> ValuesView:
        """
        Get values of the group.

        Returns:
            The values of the group.
        """
        return self.group.values()

    def items(self) -> AbstractSet:
        """
        Get (key, value) tuples of the group.

        Returns:
            The items of the group.
        """
        return self.group.items()

    def attrs(self, key: str) -> Any:
        """
        Get an attribute, identified by a key, from the group.

        Args:
            key: the name of the attribute.

        Returns:
            The attribute identified by the key, from the group.
        """
        # get attribute from group
        attribute = self.group.attrs[key]

        return self._check_type(attribute)

    @staticmethod
    def _check_type(data: Any) -> Any:
        """
        Convert data into the expected types.

        Args:
            data: any object which type should be checked.
        """
        # if attribute is an array of bytes, convert bytes to strings
        if isinstance(data, (np.ndarray, np.generic)) and data.dtype.type is np.bytes_:
            return data.astype(np.str_)

        elif isinstance(data, np.ndarray) and data.ndim == 0:
            if data.dtype.type is np.int_:
                return int(data)

            elif data.dtype.type is np.float_:
                return float(data)

            elif data.dtype.type is np.str_ or data.dtype.type is np.object_:
                return str(data)

            elif data.dtype.type is np.bool_:
                return bool(data)

        return data

    def isinstance(self, _type: type) -> bool:
        return isinstance(self.group, _type)

    def is_string(self) -> bool:
        return self.group.dtype == 'object'

    def as_string(self, encoding: Literal['UTF-8', 'ASCII'] = 'UTF-8') -> str:
        if not self.is_string():
            raise TypeError('Cannot convert non-string H5GroupReader to a string.')

        return self.group.asstr(encoding=encoding)[()]

    def create_group(self,
                     name: str,
                     track_order: Any = None) -> None:
        self.group.create_group(name=name, track_order=track_order)


def parse_path(path: None | str | Path) -> None | Path:
    """
    Convert a given path to a valid path. The '~' character is replaced by the $HOME variable.

    Args:
        path: a path to parse.

    Returns:
        A valid path.
    """
    if path is None:
        return None

    # make sure directory is a path
    if not isinstance(path, Path):
        path = Path(path)

    if path.parts[0] == '~':
        path = Path(os.environ['HOME'] / Path("/".join(path.parts[1:])))

    return path
