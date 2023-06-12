from __future__ import annotations

from typing import Iterator, MutableMapping

import vdata
from vdata.timepoint import TimePoint
from vdata.vdataframe import VDataFrame


class TimeDict(MutableMapping[str, VDataFrame]):
        
    # region magic methods
    def __init__(self,
                 vdata: vdata.VData,
                 **kwargs: VDataFrame):
        self._vdata = vdata
        self._dict = kwargs

    def __getitem__(self, key: str) -> VDataFrame:
        return self._dict[key]

    def get_timepoint(self, key: str, timepoint: TimePoint | str) -> VDataFrame:
        index = self._vdata.obs.index_at(timepoint)
        return self._dict[key].loc[index, index]

    def __setitem__(self,
                    key: str,
                    value: VDataFrame) -> None:
        self._dict[key] = value
        
    def __delitem__(self, key: str) -> None:
        del self._dict[key]
    
    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    # endregion
