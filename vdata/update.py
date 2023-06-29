import pickle
from pathlib import Path
from typing import Any, cast

import ch5mpy as ch
import numpy as np

from vdata import TemporalDataFrame, VDataFrame


def update_tdf(data: ch.File) -> None:
    data.attrs["__h5_type__"] = "object"
    data.attrs["__h5_class__"] = np.void(pickle.dumps(TemporalDataFrame, protocol=pickle.HIGHEST_PROTOCOL))
    del data.attrs["type"]

    if data.attrs["timepoints_column_name"] == "__ATTRIBUTE_None__":
        data.attrs["timepoints_column_name"] = "__h5_NONE__"

    data.move("timepoints", "timepoints_array")
    data.move("values_numerical", "numerical_array")
    data.move("values_string", "string_array")

    for array_data in data.values():
        array_data = cast(ch.Dataset[Any], array_data)
        if array_data.dtype == object:
            array_data.attrs["dtype"] = "str"


def _update_vdf(data: ch.File) -> None:
    data.attrs["__h5_type__"] = "object"
    data.attrs["__h5_class__"] = np.void(pickle.dumps(VDataFrame, protocol=pickle.HIGHEST_PROTOCOL))
    del data.attrs["type"]

    columns = []

    if "data_numeric" in data.keys():
        data.move("data_numeric/data", "data_numeric*")

        if data["data_numeric"]["columns"].dtype == object:
            columns += list(data["data_numeric"]["columns"].asstr())
        else:
            columns += list(data["data_numeric"]["columns"])

        del data["data_numeric"]
        data.move("data_numeric*", "data_numeric")

    else:
        data.create_dataset("data_numeric", data=np.empty((len(data["index"]), 0)))

    if "data_str" in data.keys():
        data.move("data_str/data", "data_string")

        if data["data_str"]["columns"].dtype == object:
            columns += list(data["data_str"]["columns"].asstr())
        else:
            columns += list(data["data_str"]["columns"])

        del data["data_str"]

    else:
        data.create_dataset("data_string", data=np.empty((len(data["index"]), 0)))

    data["columns"][()] = columns
    data.create_dataset("columns_stored_order", data=data["columns"])

    for array_data in data.values():
        array_data = cast(ch.Dataset[Any], array_data)
        if array_data.dtype == object:
            array_data.attrs["dtype"] = "str"


def update_vdata(path: Path | str) -> None:
    data = ch.File(path, mode=ch.H5Mode.READ_WRITE)

    # layers ------------------------------------------------------------------
    for layer in data["layers"].values():
        update_tdf(layer)

    # obs ---------------------------------------------------------------------
    if "obs" not in data.keys():
        first_layer = data["layers"][list(data["layers"].keys())[0]]

        obs = TemporalDataFrame(
            index=ch.read_object(first_layer["index"]),
            repeating_index=first_layer.attrs["repeating_index"],
            time_list=ch.read_object(first_layer["timepoints_array"]),
        )
        ch.write_object(data, "obs", obs)
    else:
        update_tdf(data["obs"])

    for obsm_tdf in data["obsm"].values():
        update_tdf(obsm_tdf)

    for obsp_vdf in data["obsp"].values():
        _update_vdf(obsp_vdf)

    # var ---------------------------------------------------------------------
    if "var" not in data.keys():
        first_layer = data["layers"][list(data["layers"].keys())[0]]

        var = VDataFrame(
            index=np.concatenate(
                (ch.read_object(first_layer["columns_numerical"]), ch.read_object(first_layer["columns_string"]))
            )
        )
        ch.write_object(data, "var", var)
    else:
        _update_vdf(data["var"])

    for varm_vdf in data["varm"].values():
        _update_vdf(varm_vdf)

    for varp_vdf in data["varp"].values():
        _update_vdf(varp_vdf)

    # timepoints --------------------------------------------------------------
    if "timepoints" not in data.keys():
        first_layer = data["layers"][list(data["layers"].keys())[0]]

        timepoints = VDataFrame({"value": np.unique(ch.read_object(first_layer["timepoints_array"]))})
        ch.write_object(data, "timepoints", timepoints)
    else:
        _update_vdf(data["timepoints"])

    data.close()


# from vdata.update import update_vdata
# update_vdata(output_dir / "vdata.vd")
