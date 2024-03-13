from __future__ import annotations

from typing import Any

import ch5mpy as ch
import numpy as np


def _update_array_v0_to_v1(arr: ch.Dataset[Any]) -> None:
    if arr.dtype == object or np.issubdtype(arr.dtype, bytes):
        arr.attributes["dtype"] = "str"


def _update_array_v1_to_v2(arr: ch.Dataset[Any]) -> None:
    pass


update_array = {
    0: _update_array_v0_to_v1,
    1: _update_array_v1_to_v2,
}
