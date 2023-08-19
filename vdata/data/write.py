from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ch5mpy as ch
from tqdm.auto import tqdm

from vdata.data._file import NoData
from vdata.IO.logger import generalLogger
from vdata.update import CURRENT_VERSION
from vdata.utils import spacer

if TYPE_CHECKING:
    from vdata.data.vdata import VData
    from vdata.data.view import VDataView


def write_vdata_in_h5dict(data: VData | VDataView, values: ch.H5Dict[Any], verbose: bool = True) -> None:
    nb_items_to_write = (
        len(data.layers) + len(data.obsm) + len(data.obsp) + len(data.varm) + len(data.varp) + len(data.uns) + 9
    )
    progressBar = tqdm(total=nb_items_to_write, desc=f"writing VData {data.name}", unit="object") if verbose else None

    values.attributes.set(name=data.name, __vdata_write_version__=CURRENT_VERSION)

    ch.write_objects(
        values,
        progress=progressBar,
        layers=data.layers.data,
        obs=data.obs,
        obsm=data.obsm.data,
        obsp=data.obsp.data,
        var=data.var,
        varm=data.varm.data,
        varp=data.varp.data,
        timepoints=data.timepoints,
        uns=data.uns,
    )

    if progressBar is not None:
        progressBar.close()


def write_vdata(data: VData | VDataView, file: str | Path | None = None, verbose: bool = True) -> ch.H5Dict[Any]:
    """
    Save a VData object in HDF5 file format.

    Args:
        file: path to save the VData
        verbose: print a progress bar while saving objects in this VData ? (default: True)
    """
    if data.data is not NoData._:
        if data.data.mode == ch.H5Mode.READ_WRITE:
            raise NotImplementedError("Should not be necessary")

        raise ValueError("Cannot save backed VData in 'r' mode !")

    if file is None:
        raise ValueError("No file path was provided for writing this VData.")

    file = Path(file).expanduser()
    if file.suffix != ".vd":
        generalLogger.warning("Invalid file suffix, it has been changed to '.vd'.")
        file = file.with_suffix(".vd")

    # make sure the path exists
    file.parent.mkdir(parents=True, exist_ok=True)

    # TODO : better handling of already existing files
    if file.exists():
        file.unlink()

    h5_data = ch.H5Dict.read(file, mode=ch.H5Mode.READ_WRITE_CREATE)

    write_vdata_in_h5dict(data, h5_data, verbose=verbose)

    return h5_data


def write_vdata_to_csv(
    data: VData | VDataView,
    directory: str | Path,
    sep: str = ",",
    na_rep: str = "",
    index: bool = True,
    header: bool = True,
) -> None:
    """
    Save layers, timepoints, obs, obsm, obsp, var, varm and varp to csv files in a directory.

    Args:
        directory: path to a directory for saving the matrices
        sep: delimiter character
        na_rep: string to replace NAs
        index: write row names ?
        header: Write col names ?
    """
    directory = Path(directory).expanduser()

    # make sure the directory exists and is empty
    directory.mkdir(parents=True, exist_ok=True)

    if len(list(directory.iterdir())):
        raise IOError("The directory is not empty.")

    # save metadata
    with open(directory / ".metadata.json", "w") as metadata:
        json.dump(
            {
                "obs": {"timepoints_column_name": data.obs.timepoints_column_name},
                "obsm": {
                    obsm_TDF_name: {"timepoints_column_name": obsm_TDF.get_timepoints_column_name()}
                    for obsm_TDF_name, obsm_TDF in data.obsm.items()
                },
                "layers": {
                    layer_TDF_name: {"timepoints_column_name": layer_TDF.get_timepoints_column_name()}
                    for layer_TDF_name, layer_TDF in data.layers.items()
                },
            },
            metadata,
        )

    # save matrices
    generalLogger.info(lambda: f"{spacer(1)}Saving TemporalDataFrame obs")
    data.obs.to_csv(directory / "obs.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(lambda: f"{spacer(1)}Saving TemporalDataFrame var")
    data.var.to_csv(directory / "var.csv", sep, na_rep, index=index, header=header)
    generalLogger.info(lambda: f"{spacer(1)}Saving TemporalDataFrame time-points")
    data.timepoints.to_csv(directory / "timepoints.csv", sep, na_rep, index=index, header=header)

    for dataset in (data.layers, data.obsm, data.obsp, data.varm, data.varp):
        generalLogger.info(lambda: f"{spacer(1)}Saving {dataset.name}")
        dataset.to_csv(directory, sep, na_rep, index, header, spacer=spacer(2))

    if len(data.uns):
        generalLogger.warning(lambda: f"'uns' data stored in VData '{data.name}' cannot be saved to a csv.")
