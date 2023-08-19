from typing import Collection

import numpy as np

from vdata._typing import IFS


class Index:

    __slots__ = "values", "is_repeating"

    # region magic methods
    def __init__(self, values: Collection[IFS], repeats: int = 1):
        self.values = np.tile(np.array(values), repeats)
        self.is_repeating = repeats > 1

        if repeats == 1 and len(self.values) != len(np.unique(self.values)):
            raise ValueError("Index values must be all unique if not repeating.")

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"Index({self.values}, repeating={self.is_repeating})"

    # endregion
