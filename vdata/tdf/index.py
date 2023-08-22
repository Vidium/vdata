from typing import Any, Collection, cast

import numpy as np
import numpy.typing as npt

from vdata._typing import IFS


class Index:

    __slots__ = "values", "repeats"

    # region magic methods
    def __init__(self, values: Collection[IFS], repeats: int = 1):
        self.values = np.tile(np.array(values), repeats)
        self.repeats = repeats

        if repeats == 1 and len(self.values) != len(np.unique(self.values)):
            raise ValueError("Index values must be all unique if not repeating.")

    def __repr__(self) -> str:
        return f"Index({self.values}, repeating={self.is_repeating})"

    def __getitem__(self, item: Any) -> Any:
        return self.values[item]

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool | npt.NDArray[np.bool_]:  # type: ignore[override]
        if isinstance(other, Index):
            return np.array_equal(self.values, other.values) and self.is_repeating == other.is_repeating

        return cast(npt.NDArray[np.bool_], self.values == other)

    def __hash__(self) -> int:
        return hash(self.values.data.tobytes()) + int(self.is_repeating)

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        if dtype is None:
            return self.values

        return self.values.astype(dtype)

    # endregion

    # region attributes
    @property
    def is_repeating(self) -> bool:
        return self.repeats > 1

    # endregion
