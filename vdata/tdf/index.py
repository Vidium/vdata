from __future__ import annotations

from typing import Collection, Hashable, cast

import numpy as np
import pandas as pd
from pandas._libs.internals import BlockValuesRefs

from vdata._typing import IFS


class RepeatingIndex(pd.Index):
    _metadata = ["repeats"]

    # region magic methods
    def __new__(
        cls,
        data: Collection[IFS],
        repeats: int = 1,
    ) -> RepeatingIndex:
        data = np.tile(np.array(data), repeats)
        return cast(RepeatingIndex, cls._simple_new(data, None, repeats=repeats))

    @classmethod
    def _simple_new(
        cls: type[RepeatingIndex], values: np.ndarray, name: Hashable, refs=None, repeats=None
    ) -> RepeatingIndex | pd.Index:
        if repeats is None:
            return pd.Index._simple_new(values, name, refs)
        assert isinstance(values, cls._data_cls), type(values)

        if repeats == 1 and len(values) != len(np.unique(values)):
            raise ValueError("Index values must be all unique if not repeating.")

        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        if refs is not None:
            result._references = refs
        else:
            result._references = BlockValuesRefs()
        result._references.add_index_reference(result)

        result.repeats = repeats

        return result

    def __hash__(self) -> int:
        return hash(self.values.data.tobytes()) + int(self.is_repeating)

    # endregion

    # region attributes
    @property
    def _constructor(self) -> type[pd.Index]:
        return pd.Index

    @property
    def is_repeating(self) -> bool:
        return self.repeats > 1

    # endregion

    # region methods
    def _format_attrs(self) -> list[tuple[str, str | int | bool | None]]:
        attrs = super()._format_attrs()
        attrs.append(("repeating", self.is_repeating))
        return attrs

    # endregion
