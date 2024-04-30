import numpy as np
import pandas as pd

from vdata.tdf.index import RepeatingIndex


def test_repeating_index_isin():
    index = RepeatingIndex([1, 2, 3])

    assert np.array_equal(pd.Index([2, 4]).isin(index), [True, False])


def test_repeating_index_isin_large():
    index = RepeatingIndex(list(range(15, 404)))

    assert np.array_equal(
        pd.Index(list(range(389))).isin(index), [False for _ in range(15)] + [True for _ in range(374)]
    )
