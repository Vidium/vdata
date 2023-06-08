from typing import Any

import numpy as np
import numpy.typing as npt

from vdata.array_view import NDArrayView


class Container:
    def __init__(self, arr: npt.NDArray[Any]) -> None:
        self.arr = arr


def test_array_view_conversion_to_numpy_has_correct_dtype() -> None:
    arr = np.array([['abcd', 'efgh'], ['ijkl', 'lmno']])
    v_arr: NDArrayView = NDArrayView(Container(arr), 'arr', slice(None))
    new_arr = np.array(v_arr)
    
    assert new_arr.dtype == np.dtype('<U4')
    