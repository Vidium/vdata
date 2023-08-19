from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from vdata.array_view import ArrayGetter, NDArrayView


@dataclass
class Container:
    array: npt.NDArray[Any]


def test_array_view_conversion_to_numpy_has_correct_dtype() -> None:
    data = Container(np.array([["abcd", "efgh"], ["ijkl", "lmno"]]))
    v_arr: NDArrayView = NDArrayView(ArrayGetter(data, "array"), slice(None))
    new_arr = np.array(v_arr)

    assert new_arr.dtype == np.dtype("<U4")
