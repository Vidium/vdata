from __future__ import annotations

from pathlib import Path
from typing import Any

import ch5mpy as ch
import numpy as np
from h5dataframe import H5DataFrame
from tqdm.auto import tqdm

import vdata
from vdata.update.dict import update_dict
from vdata.update.tdf import update_tdf
from vdata.update.vdf import update_vdf

CURRENT_VERSION = 2


class NoBar:
    def update(self) -> None:
        pass

    def close(self) -> None:
        pass


def _update_vdata(data: ch.H5Dict[Any], from_version: int, progressBar: tqdm[Any] | NoBar) -> None:
    # layers ------------------------------------------------------------------
    for layer in (data @ "layers").keys():
        update_tdf[from_version]((data @ "layers") @ layer)
        progressBar.update()

    # obs ---------------------------------------------------------------------
    if "obs" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        obs = vdata.TemporalDataFrame(
            index=ch.read_object(first_layer["index"]),
            # repeating_index=first_layer.attrs["repeating_index"],
            timepoints=ch.read_object(first_layer["timepoints_array"]),
        )
        ch.write_object(obs, data, "obs")
    else:
        update_tdf[from_version](data @ "obs")

    progressBar.update()

    for obsm_tdf in (data @ "obsm").keys():
        update_tdf[from_version]((data @ "obsm") @ obsm_tdf)
        progressBar.update()

    for obsp_vdf in (data @ "obsp").keys():
        update_vdf[from_version]((data @ "obsp") @ obsp_vdf)
        progressBar.update()

    # var ---------------------------------------------------------------------
    if "var" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        var = H5DataFrame(
            index=np.concatenate(
                (ch.read_object(first_layer["columns_numerical"]), ch.read_object(first_layer["columns_string"]))
            )
        )
        ch.write_object(var, data, "var")
    else:
        update_vdf[from_version](data @ "var")

    progressBar.update()

    for varm_vdf in (data @ "varm").values():
        update_vdf[from_version](varm_vdf)
        progressBar.update()

    for varp_vdf in (data @ "varp").values():
        update_vdf[from_version](varp_vdf)
        progressBar.update()

    # timepoints --------------------------------------------------------------
    if "timepoints" not in data.keys():
        first_layer = (data @ "layers")[list((data @ "layers").keys())[0]]

        timepoints = H5DataFrame({"value": np.unique(ch.read_object(first_layer["timepoints_array"]))})
        ch.write_object(timepoints, data, "timepoints")
    else:
        update_vdf[from_version](data @ "timepoints")

    progressBar.update()

    # uns ---------------------------------------------------------------------
    if "uns" not in data.keys():
        data["uns"] = {}

    else:
        update_dict[from_version](data @ "uns")

    progressBar.update()


def update_vdata(data: Path | str | ch.H5Dict[Any], verbose: bool = True) -> int:
    """
    Update an h5 file containing a vdata saved in an older version.

    Args:
        data: path to the h5 file to update.
        verbose: print a progress bar ? (default: True)
    """
    _was_opened_here = not isinstance(data, ch.H5Dict)
    if not isinstance(data, ch.H5Dict):
        data = ch.H5Dict.read(data, mode=ch.H5Mode.READ_WRITE)

    data_version = data.attributes.get("__vdata_write_version__", 0)
    if data_version == CURRENT_VERSION:
        return data_version

    if data_version > CURRENT_VERSION:
        raise ValueError(
            f"VData object was written with a version ({data_version}) of the write protocol higher than the current \
                    one ({CURRENT_VERSION})"
        )

    if not ch.H5Mode.has_write_intent(data.file.file.mode):
        raise IOError(
            "Cannot update VData file to current version because it was opened with no write intent. \
                 Please open it in READ_WRITE mode (r+)."
        )

    nb_items_to_write = (
        4 + len(data @ "layers") + len(data @ "obsm") + len(data @ "obsp") + len(data @ "varm") + len(data @ "varp")
    )
    for v in range(data_version, CURRENT_VERSION):
        progressBar: tqdm[Any] | NoBar = (
            tqdm(total=nb_items_to_write, desc=f" Updating old VData file [version {v} to {v+1}]", unit="object")
            if verbose
            else NoBar()
        )

        _update_vdata(data, v, progressBar)
        data.attributes["__vdata_write_version__"] = v + 1

        progressBar.close()

    if _was_opened_here:
        data.close()

    return data_version
