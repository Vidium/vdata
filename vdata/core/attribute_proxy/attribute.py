# coding: utf-8
# Created on 23/10/2022 10:16
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from typing import Union

from vdata.core.tdf.name_utils import H5Data


# ====================================================
# code
NONE_VALUE = '__ATTRIBUTE_None__'
VALUE = Union[str, int, float, bool]


class AttributeProxy:
    """Simple proxy for managing attributes in h5py files."""

    # region magic methods
    def __init__(self,
                 file: H5Data):
        self._file = file

    def __repr__(self) -> str:
        return f"AttributeProxy on file '{self._file.file.filename}{self._file.name}'"

    def __getitem__(self,
                    item: str) -> VALUE | None:
        value = self._file.attrs[item]
        return None if value == NONE_VALUE else value

    def __setitem__(self,
                    key: str,
                    value: VALUE | None) -> None:
        if value is None:
            self._file.attrs[key] = NONE_VALUE

        else:
            self._file.attrs[key] = value

    # endregion
