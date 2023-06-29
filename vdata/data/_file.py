from __future__ import annotations

from enum import Enum
from typing import Any, NoReturn


class _NoDataType:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> _NoDataType:
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def close(self) -> NoReturn:
        raise ValueError("Cannot close object which is not backed on h5 file.")


class NoData(Enum):
    _ = _NoDataType()
