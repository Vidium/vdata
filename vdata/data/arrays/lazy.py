from dataclasses import dataclass

import pandas as pd
from h5dataframe import H5DataFrame

from vdata._typing import NDArray_IFS


@dataclass
class LazyLoc:
    h5df: H5DataFrame
    loc: NDArray_IFS | tuple[NDArray_IFS, NDArray_IFS]

    # region attributes
    @property
    def shape(self) -> tuple[int, int]:
        if not isinstance(self.loc, tuple):
            return len(self.loc), self.h5df.shape[1]

        if len(self.loc) == 1:
            return len(self.loc[0]), self.h5df.shape[1]

        return len(self.loc[0]), len(self.loc[1])

    # endregion

    # region methods
    def get(self) -> H5DataFrame:
        return self.h5df.loc[self.loc]

    def copy(self) -> pd.DataFrame:
        return pd.DataFrame(self.get())

    # endregion
