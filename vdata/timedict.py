from __future__ import annotations

from typing import Iterator, MutableMapping

from h5dataframe import H5DataFrame

import vdata


class TimeDict(MutableMapping[str, H5DataFrame]):

    # region magic methods
    def __init__(self, vdata: vdata.VData, **kwargs: H5DataFrame):
        self._vdata = vdata
        self._dict = kwargs

    def __getitem__(self, key: str) -> H5DataFrame:
        return self._dict[key]

    def __setitem__(self, key: str, value: H5DataFrame) -> None:
        self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    # endregion
