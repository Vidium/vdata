from __future__ import annotations

from typing import Any, ItemsView, KeysView, MutableMapping, ValuesView

import numpy.typing as npt

from vdata.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.vdataframe import VDataFrame


class TemporalDataFrameContainerProxy:

    __slots__ = "_tdfs", "_name"

    # region magic methods
    def __init__(self, tdfs: MutableMapping[str, TemporalDataFrame | TemporalDataFrameView], name: str) -> None:
        self._tdfs = tdfs
        self._name = name

    def __repr__(self) -> str:
        return f"{self._name} with keys: {', '.join(self._tdfs.keys())}"

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        return self._tdfs[str(key)].values

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._tdfs.keys()

    def values(self) -> ValuesView[TemporalDataFrame | TemporalDataFrameView]:
        return self._tdfs.values()

    def items(self) -> ItemsView[str, TemporalDataFrame | TemporalDataFrameView]:
        return self._tdfs.items()

    # endregion


class VDataFrameContainerProxy:

    __slots__ = "_vdfs", "_name"

    # region magic methods
    def __init__(self, vdfs: MutableMapping[str, VDataFrame], name: str) -> None:
        self._vdfs = vdfs
        self._name = name

    def __repr__(self) -> str:
        return f"{self._name} with keys: {', '.join(self._vdfs.keys())}"

    # endregion
