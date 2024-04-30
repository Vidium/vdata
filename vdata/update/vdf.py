from __future__ import annotations

import pickle
from typing import Any

import ch5mpy as ch
import numpy as np
import numpy.typing as npt
from h5dataframe import H5DataFrame

from vdata.update.array import update_array


def get_common_dtype(dt1: npt.DTypeLike, dt2: npt.DTypeLike) -> type[np.generic]:
    match dt1, dt2:
        case np.int_, np.int_:
            return np.int64

        case np.float_ | np.int_, np.float_ | np.int_:
            return np.float64

        case _:
            return np.str_


def _update_vdf_v0_to_v1(data: ch.H5Dict[Any]) -> None:
    data.attributes.set(
        __h5_type__="object",
        __h5_class__=np.void(pickle.dumps(H5DataFrame, protocol=pickle.HIGHEST_PROTOCOL)),
    )
    del data.attributes["type"]

    data["arrays"] = {}

    if "data_numeric" in data.keys():
        update_array[0](data["data_numeric"]["data"])
        for col_idx, column in enumerate(data["data_numeric"]["columns"].astype(str)):
            data["arrays"][column] = data["data_numeric"]["data"][:, col_idx].flatten()

        data_num_dtype = data["data_numeric"]["columns"].dtype
        del data["data_numeric"]

    else:
        data_num_dtype = np.int64

    if "data_str" in data.keys():
        update_array[0](data["data_str"]["data"])
        for col_idx, column in enumerate(data["data_str"]["columns"].astype(str)):
            data["arrays"][column] = data["data_str"]["data"][:, col_idx].flatten()

        data_str_dtype = data["data_str"]["columns"].dtype
        del data["data_str"]

    else:
        data_str_dtype = np.int64

    del data["columns"]

    data.attributes["columns_dtype"] = get_common_dtype(data_num_dtype, data_str_dtype)

    update_array[0](data["index"])


def _update_vdf_v1_to_v2(data: ch.H5Dict[Any]) -> None:
    pass


update_vdf = {
    0: _update_vdf_v0_to_v1,
    1: _update_vdf_v1_to_v2,
}
