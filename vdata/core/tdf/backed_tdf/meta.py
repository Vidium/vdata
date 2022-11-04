# coding: utf-8
# Created on 18/10/2022 23:25
# Author : matteo

# ====================================================
# imports
from abc import ABCMeta

from typing import TYPE_CHECKING

from vdata.h5pickle.name_utils import H5Mode

if TYPE_CHECKING:
    from vdata.core.tdf.base import BaseTemporalDataFrame


# ====================================================
# code
def check_can_read(method):
    """Wrapper around BackedTemporalDataFrame methods that require access to the h5 file to check it is still open."""

    def wrapper(*args, **kwargs):
        self: BaseTemporalDataFrame = args[0]

        if self.is_closed:
            raise ValueError("Can't read TemporalDataFrame backed on closed file.")

        return method(*args, **kwargs)

    return wrapper


def check_can_write(method):
    """Wrapper around BackedTemporalDataFrame methods that need to write to the h5 file to check it is still open in
    a or r+ mode."""

    def wrapper(*args, **kwargs):
        self: BaseTemporalDataFrame = args[0]

        if self.is_closed:
            raise ValueError("Can't write to TemporalDataFrame backed on closed file.")

        if self.is_view:
            file = object.__getattribute__(self._parent, '_file')

        else:
            file = object.__getattribute__(self, '_file')

        if (m := file.file.mode) not in (H5Mode.READ_WRITE_CREATE, H5Mode.READ_WRITE):
            raise ValueError(f"Can't write to TemporalDataFrame backed on file with mode='{m}'.")

        return method(*args, **kwargs)

    return wrapper


class CheckH5File(ABCMeta):
    """Metaclass to add checks to methods of a class that need to access a h5 file for reading or writing to it."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        for attr, value in namespace.items():
            if callable(value):
                if value in kwargs.get('read', ()):
                    namespace[attr] = check_can_read(value)

                elif value in kwargs.get('write', ()):
                    namespace[attr] = check_can_write(value)

        return super().__new__(mcs, name, bases, namespace, **kwargs)
