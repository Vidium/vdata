import pickle
from pathlib import Path
from typing import Any

import ch5mpy as ch
import numpy as np
from h5dataframe import H5DataFrame
from tqdm.auto import tqdm

import vdata

CURRENT_VERSION = 1


class NoBar:
    def update(self) -> None:
        pass

    def close(self) -> None:
        pass


def _update_array(arr: ch.Dataset[Any]) -> None:
    if arr.dtype == object or np.issubdtype(arr.dtype, bytes):
        arr.attrs["dtype"] = "str"


def update_tdf(data: ch.H5Dict[Any]) -> None:
    data.attributes.set(
        __h5_type__="object",
        __h5_class__=np.void(pickle.dumps(vdata.TemporalDataFrame, protocol=pickle.HIGHEST_PROTOCOL)),
    )
    del data.attributes["type"]

    if data.attributes["timepoints_column_name"] == "__ATTRIBUTE_None__":
        data.attributes["timepoints_column_name"] = "__h5_NONE__"

    data.file.move("timepoints", "timepoints_array")
    data.file.move("values_numerical", "numerical_array")
    data.file.move("values_string", "string_array")

    for array_data in data.values():
        _update_array(array_data)


def _update_vdf(data: ch.H5Dict[Any]) -> None:
    data.attributes.set(
        __h5_type__="object", __h5_class__=np.void(pickle.dumps(H5DataFrame, protocol=pickle.HIGHEST_PROTOCOL))
    )
    del data.attributes["type"]

    data["arrays"] = {}

    if "data_numeric" in data.keys():
        _update_array(data["data_numeric"]["data"])
        for col_idx, column in enumerate(data["data_numeric"]["columns"].astype(str)):
            data["arrays"][column] = data["data_numeric"]["data"][:, col_idx].flatten()

        del data["data_numeric"]

    if "data_str" in data.keys():
        _update_array(data["data_str"]["data"])
        for col_idx, column in enumerate(data["data_str"]["columns"].astype(str)):
            data["arrays"][column] = data["data_str"]["data"][:, col_idx].flatten()

        del data["data_str"]

    del data["columns"]

    _update_array(data["index"])


def _update_dict(obj: ch.H5Dict[Any]) -> None:
    for key in obj.keys():
        if isinstance(obj @ key, ch.H5Array):
            _update_array(obj @ key)

        elif isinstance(obj @ key, ch.H5Dict):
            _update_dict(obj @ key)


def update_vdata(data: Path | str | ch.H5Dict[Any], verbose: bool = True) -> None:
    """
    Update an h5 file containing a vdata saved in an older version.

    Args:
        data: path to the h5 file to update.
        verbose: print a progress bar ? (default: True)
    """
    _was_opened_here = not isinstance(data, ch.H5Dict)
    if not isinstance(data, ch.H5Dict):
        data = ch.H5Dict.read(data, mode=ch.H5Mode.READ_WRITE)

    nb_items_to_write = (
        4 + len(data @ "layers") + len(data @ "obsm") + len(data @ "obsp") + len(data @ "varm") + len(data @ "varp")
    )
    progressBar: tqdm[Any] | NoBar = (
        tqdm(total=nb_items_to_write, desc=" Updating old VData file", unit="object") if verbose else NoBar()
    )

    # layers ------------------------------------------------------------------
    for layer in (data @ "layers").keys():
        update_tdf((data @ "layers") @ layer)
        progressBar.update()

    # obs ---------------------------------------------------------------------
    if "obs" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        obs = vdata.TemporalDataFrame(
            index=ch.read_object(first_layer["index"]),
            # repeating_index=first_layer.attrs["repeating_index"],
            timepoints=ch.read_object(first_layer["timepoints_array"]),
        )
        ch.write_object(data, "obs", obs)
    else:
        update_tdf(data @ "obs")

    progressBar.update()

    for obsm_tdf in (data @ "obsm").keys():
        update_tdf((data @ "obsm") @ obsm_tdf)
        progressBar.update()

    for obsp_vdf in (data @ "obsp").keys():
        _update_vdf((data @ "obsp") @ obsp_vdf)
        progressBar.update()

    # var ---------------------------------------------------------------------
    if "var" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        var = H5DataFrame(
            index=np.concatenate(
                (ch.read_object(first_layer["columns_numerical"]), ch.read_object(first_layer["columns_string"]))
            )
        )
        ch.write_object(data, "var", var)
    else:
        _update_vdf(data @ "var")

    progressBar.update()

    for varm_vdf in (data @ "varm").values():
        _update_vdf(varm_vdf)
        progressBar.update()

    for varp_vdf in (data @ "varp").values():
        _update_vdf(varp_vdf)
        progressBar.update()

    # timepoints --------------------------------------------------------------
    if "timepoints" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        timepoints = H5DataFrame({"value": np.unique(ch.read_object(first_layer["timepoints_array"]))})
        ch.write_object(data, "timepoints", timepoints)
    else:
        _update_vdf(data @ "timepoints")

    progressBar.update()

    # uns ---------------------------------------------------------------------
    if "uns" not in data.keys():
        data["uns"] = {}

    else:
        _update_dict(data @ "uns")

    progressBar.update()

    # -------------------------------------------------------------------------
    data.attributes["__vdata_write_version__"] = CURRENT_VERSION

    if _was_opened_here:
        data.close()
    progressBar.close()


# from vdata.update import update_vdata
# update_vdata(output_dir / "vdata.vd")
