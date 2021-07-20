# coding: utf-8
# Created on 19/01/2021 15:36
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
from pathlib import Path
from typing import Union, Tuple, AbstractSet, ValuesView, Any, Optional
from typing_extensions import Literal

from .name_utils import H5Group


# ====================================================
# code
class H5GroupReader:
    """
    Class for reading a h5py File, Group or Dataset
    """

    def __init__(self, group: 'H5Group'):
        """
        :param group: a h5py File, Group or Dataset
        """
        self.group = group

    def __getitem__(self, key: Union[str, slice, 'ellipsis', Tuple[()]]) \
            -> Union['H5GroupReader', np.ndarray, str, int, float, bool, type]:
        """
        Get a sub-group from the group, identified by a key

        :param key: the name of the sub-group
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

    def close(self) -> None:
        self.group.file.close()

    @property
    def name(self) -> str:
        """
        Get the name of the group.
        :return: the group's name.
        """
        return self.group.name

    @property
    def filename(self) -> str:
        """
        Get the filename of the group.
        :return: the group's filename.
        """
        return self.group.file.filename

    @property
    def mode(self) -> str:
        """
        Get the reading mode for the group.
        :return: the reading mode for the group.
        """
        return self.group.file.mode

    @property
    def parent(self) -> 'H5GroupReader':
        """
        Get the parent H5GroupReader.
        :return: the parent H5GroupReader.
        """
        return H5GroupReader(self.group.parent)

    def keys(self) -> AbstractSet:
        """
        Get keys of the group.
        :return: the keys of the group.
        """
        return self.group.keys()

    def values(self) -> ValuesView:
        """
        Get values of the group.
        :return: the values of the group.
        """
        return self.group.values()

    def items(self) -> AbstractSet:
        """
        Get (key, value) tuples of the group.
        :return: the items of the group.
        """
        return self.group.items()

    def attrs(self, key: str) -> Any:
        """
        Get an attribute, identified by a key, from the group.

        :param key: the name of the attribute.
        :return: the attribute identified by the key, from the group.
        """
        # get attribute from group
        attribute = self.group.attrs[key]

        return self._check_type(attribute)

    @staticmethod
    def _check_type(data: Any) -> Any:
        """
        Convert data into the expected types.

        :param data: any object which type should be checked.
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

    def isstring(self) -> bool:
        return self.group.dtype == 'object'

    def asstring(self, encoding: Literal['UTF-8', 'ASCII'] = 'UTF-8') -> bool:
        if not self.isstring():
            raise TypeError('Cannot convert non-string H5GroupReader to a string.')

        return self.group.asstr(encoding=encoding)[()]


def parse_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
    """
    Convert a given path to a valid path. The '~' character is replaced by the $HOME variable.

    :param path: a path to parse.
    :return: a valid path.
    """
    if path is None:
        return None

    # make sure directory is a path
    if not isinstance(path, Path):
        path = Path(path)

    if path.parts[0] == '~':
        path = Path(os.environ['HOME'] / Path("/".join(path.parts[1:])))

    return path
