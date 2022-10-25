# coding: utf-8
# Created on 25/10/2022 09:06
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import Self


# ====================================================
# code
class DTypeMeta(type):
    def __new__(mcs,
                name,
                bases,
                namespace):
        obj = super().__new__(mcs, name, bases, namespace)
        return obj

    def __repr__(cls) -> str:
        return f"dataset_proxy.dtype[{cls.__name__}]"


class DType(metaclass=DTypeMeta):
    """Data type for dataset proxies."""


class Generic(DType):

    def __repr__(self) -> str:
        return f"dataset_proxy.dtype('{self.__class__.__name__}')"

    def __call__(self,
                 value: Any) -> Any:
        return self._cast(value)

    @property
    def type(self) -> Self:
        return self

    def _cast(self,
              value: Any) -> Any:
        raise NotImplementedError


class Num_(Generic):
    """Numeric dtype for dataset proxies."""


class Int_(Num_):
    """Integer dtype for dataset proxies."""

    def _cast(self, value) -> Any:
        return int(value)


class Float_(Num_):
    """Integer dtype for dataset proxies."""

    def _cast(self, value) -> Any:
        return float(value)


class Str_(Generic):
    """String dtype for dataset proxies."""

    def _cast(self, value) -> Any:
        return str(value)


class TP_(Generic):
    """TimePoint dtype for dataset proxies."""

    def _cast(self, value) -> Any:
        from vdata import TimePoint
        return TimePoint(value)


num_ = Num_()
int_ = Int_()
float_ = Float_()
str_ = Str_()
tp_ = TP_()

DTYPE_TO_NP = {
    num_: np.float_,
    int_: np.int_,
    float_: np.float_
}


def issubdtype(typ: DType,
               other: DType | tuple[DType]) -> bool:
    if isinstance(other, tuple):
        return any((issubdtype(typ, o) for o in other))

    return isinstance(typ, type(other))
