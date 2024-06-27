from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Collection, Iterable, Literal, overload

import ch5mpy as ch
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from h5dataframe import H5DataFrame
from tqdm.auto import tqdm

import vdata
import vdata.timepoint as tp
from vdata._typing import IFS
from vdata.array_view import NDArrayView
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrame
from vdata.update.update import CURRENT_VERSION
from vdata.utils import repr_array


@overload
def convert_vdata_to_anndata(
    data: vdata.VData,
    into_one: Literal[True] = True,
    timepoints_list: str | tp.TimePoint | Collection[str | tp.TimePoint] | None = None,
    with_timepoints_column: bool = True,
    layer_as_X: str | None = None,
    layers_to_export: list[str] | None = None,
) -> AnnData: ...
@overload
def convert_vdata_to_anndata(
    data: vdata.VData,
    into_one: Literal[False],
    timepoints_list: str | tp.TimePoint | Collection[str | tp.TimePoint] | None = None,
    with_timepoints_column: bool = True,
    layer_as_X: str | None = None,
    layers_to_export: list[str] | None = None,
) -> list[AnnData]: ...
def convert_vdata_to_anndata(
    data: vdata.VData,
    into_one: bool = True,
    timepoints_list: str | tp.TimePoint | Collection[str | tp.TimePoint] | None = None,
    with_timepoints_column: bool = True,
    layer_as_X: str | None = None,
    layers_to_export: list[str] | None = None,
) -> AnnData | list[AnnData]:
    """
    Convert a VData object to an AnnData object.

    Args:
        timepoints_list: a list of time points for which to extract data to build the AnnData. If set to
            None, all timepoints are selected.
        into_one: Build one AnnData, concatenating the data for multiple time points (True), or build one
            AnnData for each time point (False) ?
        with_timepoints_column: store time points data in the obs DataFrame. This is only used when
            concatenating the data into a single AnnData (i.e. into_one=True).
        layer_as_X: name of the layer to use as the X matrix. By default, the first layer is used.
        layers_to_export: if None export all layers

    Returns:
        An AnnData object with data for selected time points.
    """
    # TODO : obsp is not passed to AnnData

    generalLogger.debug(
        "\u23be VData conversion to AnnData : begin " "---------------------------------------------------------- "
    )

    if timepoints_list is None:
        _timepoints_list: tp.TimePointArray | NDArrayView[tp.TimePoint] = data.timepoints_values

    else:
        _timepoints_list = tp.as_timepointarray(timepoints_list)
        _timepoints_list = tp.atleast_1d(_timepoints_list[np.where(np.in1d(_timepoints_list, data.timepoints_values))])

    generalLogger.debug(lambda: f"Selected time points are : {repr_array(_timepoints_list)}")

    if into_one:
        return _convert_vdata_into_one_anndata(
            data, with_timepoints_column, _timepoints_list, layer_as_X, layers_to_export
        )

    return _convert_vdata_into_many_anndatas(data, _timepoints_list, layer_as_X)


def _convert_vdata_into_one_anndata(
    data: vdata.VData,
    with_timepoints_column: bool,
    timepoints_list: tp.TimePointArray | NDArrayView[tp.TimePoint],
    layer_as_X: str | None,
    layers_to_export: Iterable[str] | None,
) -> AnnData:
    generalLogger.debug("Convert to one AnnData object.")

    tp_col_name = data.obs.get_timepoints_column_name() if with_timepoints_column else None

    view = data[timepoints_list]
    if layer_as_X is None:
        layer_as_X = list(view.layers.keys())[0]

    elif layer_as_X not in view.layers.keys():
        raise ValueError(f"Layer '{layer_as_X}' was not found.")

    X = view.layers[layer_as_X].to_pandas()
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    if layers_to_export is None:
        layers_to_export = view.layers.keys()

    anndata = AnnData(
        X=X,
        layers={key: view.layers[key].to_pandas(str_index=True) for key in layers_to_export},
        obs=view.obs.to_pandas(with_timepoints=tp_col_name, str_index=True),
        obsm={key: arr.values_num for key, arr in view.obsm.items()},
        obsp={key: arr.values for key, arr in view.obsp.items()},
        var=view.var.copy(),
        varm=view.varm.dict_copy(),
        varp=view.varp.dict_copy(),
        uns=view.uns.copy(),
    )

    generalLogger.debug(
        "\u23bf VData conversion to AnnData : end " "---------------------------------------------------------- "
    )

    return anndata


def _convert_vdata_into_many_anndatas(
    data: vdata.VData,
    timepoints_list: tp.TimePointArray | NDArrayView[tp.TimePoint],
    layer_as_X: str | None,
) -> list[AnnData]:
    generalLogger.debug("Convert to many AnnData objects.")

    result = []
    for time_point in timepoints_list:
        view = data[time_point]

        if layer_as_X is None:
            layer_as_X = list(view.layers.keys())[0]

        elif layer_as_X not in view.layers.keys():
            raise ValueError(f"Layer '{layer_as_X}' was not found.")

        X = view.layers[layer_as_X].to_pandas()
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)

        result.append(
            AnnData(
                X=X,
                layers={key: layer.to_pandas(str_index=True) for key, layer in view.layers.items()},
                obs=view.obs.to_pandas(str_index=True),
                obsm={key: arr.to_pandas(str_index=True) for key, arr in view.obsm.items()},
                var=view.var.copy(),
                varm={key: arr.copy() for key, arr in view.varm.items()},
                varp={key: arr.copy() for key, arr in view.varp.items()},
                uns=view.uns,
            )
        )

    generalLogger.debug(
        "\u23bf VData conversion to AnnData : end " "---------------------------------------------------------- "
    )

    return result


def not_categorical(obj: ch.H5Array[Any] | ch.H5Dict) -> ch.H5Array[Any]:
    if isinstance(obj, ch.H5Array):
        return obj

    if "categories" not in obj.keys():
        raise ValueError("Cannot convert object")

    flat = obj["categories"][obj["codes"]]
    if flat.dtype == np.dtype(object):
        flat = flat.astype("str")

    return flat


def convert_anndata_to_vdata(
    path: Path | str,
    timepoint: IFS | tp.TimePoint = tp.TimePoint("0h"),
    timepoints_column_name: str | None = None,
    inplace: bool = False,
    drop_X: bool = False,
) -> ch.H5Dict[Any]:
    r"""
    Convert an anndata h5 file into a valid vdata h5 file.
    /!\ WARNING : if done inplace, you won't be able to open the file as an anndata anymore !

    Args:
        path: path to the anndata h5 file to convert.
        timepoint: a unique timepoint to set for the data in the anndata.
        timepoints_column_name: the name of the column in anndata's obs to use as indicator of time point for the data.
        inplace: perform file conversion directly on the anndata h5 file ? (default: False)
        drop_X: do not preserve the 'X' dataset ? (default: False)
    """
    path = Path(path)

    if not inplace:
        print("Copying file ...")
        # TODO maybe implement https://stackoverflow.com/a/49684845
        shutil.copy(path, path.with_suffix(".vd"))

    else:
        shutil.move(path, path.with_suffix(".vd"))

    data = ch.H5Dict.read(path.with_suffix(".vd"), mode=ch.H5Mode.READ_WRITE)
    data.attributes["__vdata_write_version__"] = CURRENT_VERSION

    # X -----------------------------------------------------------------------
    if not drop_X:
        data["layers"]["X"] = data["X"]

    del data["X"]

    # progress bar ------------------------------------------------------------
    nb_items_to_write = (
        3
        + len(data["layers"])
        + len(data["obsm"])
        + len(data["obsp"])
        + len(data["varm"])
        + len(data["varp"])
        + int(not drop_X)
    )
    progressBar = tqdm(total=nb_items_to_write, desc="Converting anndata to VData", unit="object")

    # timepoints --------------------------------------------------------------
    timepoint = tp.TimePoint(timepoint)

    if timepoints_column_name is not None:
        if timepoints_column_name not in data["obs"]:
            raise ValueError(f"Could not find column '{timepoints_column_name}' in obs columns.")

        timepoints_list = tp.as_timepointarray(not_categorical(data["obs"][timepoints_column_name]))

    else:
        timepoints_list = tp.TimePointArray(
            np.ones(data["obs"][next(iter(data["obs"]))].shape[0]) * timepoint.value, unit=timepoint.unit
        )

    ch.write_object(H5DataFrame({"value": np.unique(timepoints_list, equal_nan=False)}), data, "timepoints")
    progressBar.update()

    # obs ---------------------------------------------------------------------
    _obs_index = data["obs"]["_index"].astype(str)
    del data["obs"]["_index"]

    obs_data = {k: not_categorical(v) for k, v in data["obs"].items()}

    del data["obs"]

    ch.write_object(
        TemporalDataFrame(obs_data, index=_obs_index, timepoints=timepoints_list, lock=(True, False), name="obs"),
        data,
        "obs",
    )

    progressBar.update()

    # var ---------------------------------------------------------------------
    _var_index = data["var"]["_index"].astype(str)

    var_data = data["var"].copy()
    del var_data["_index"]

    del data["var"]

    ch.write_object(H5DataFrame(var_data, index=_var_index), data, "var")

    progressBar.update()

    # layers ------------------------------------------------------------------
    for layer_name, layer_data in data["layers"].items():
        if layer_name == "X" and isinstance(layer_data, ch.H5Dict):
            raise TypeError("Cannot convert X layer if it is a sparse matrix.")

        if layer_data.attributes.get("encoding-type", "") == "csr_matrix":
            matrix_data = layer_data["data"]
            col = layer_data["indices"]
            # FIXME: implement np.ediff1d on H5Arrays
            row = np.repeat(np.arange(len(layer_data["indptr"]) - 1), np.ediff1d(layer_data["indptr"].copy()))

            # FIXME: proper handling of sparse matrices
            layer_data = sp.csr_matrix((matrix_data, (row, col)), shape=layer_data.attributes["shape"]).toarray()

        else:
            layer_data = layer_data.copy()

        del data["layers"][layer_name]

        ch.write_object(
            TemporalDataFrame(
                layer_data,
                index=_obs_index,
                columns=_var_index,
                timepoints=timepoints_list,
                lock=(True, True),
                name=layer_name,
            ),
            data["layers"],
            layer_name,
        )

        progressBar.update()

    # obsm --------------------------------------------------------------------
    for array_name, array_data in data["obsm"].items():
        array_data = array_data.copy()
        del data["obsm"][array_name]

        ch.write_object(
            TemporalDataFrame(
                array_data, index=_obs_index, timepoints=timepoints_list, lock=(True, False), name=array_name
            ),
            data["obsm"],
            array_name,
        )

        progressBar.update()

    # obsp --------------------------------------------------------------------
    for array_name, array_data in data["obsp"].items():
        array_data = array_data.copy()
        del data["obsp"][array_name]

        ch.write_object(H5DataFrame(array_data, index=_obs_index, columns=_obs_index), data, "timepoints")

        progressBar.update()

    # varm --------------------------------------------------------------------
    for array_name, array_data in data["varm"].items():
        array_data = array_data.copy()
        del data["varm"][array_name]

        ch.write_object(H5DataFrame(array_data, index=_var_index), data, "timepoints")

        progressBar.update()

    # varp --------------------------------------------------------------------
    for array_name, array_data in data["varp"].items():
        array_data = array_data.copy()
        del data["varp"][array_name]

        ch.write_object(H5DataFrame(array_data, index=_var_index, columns=_var_index), data, "timepoints")

        progressBar.update()

    progressBar.close()

    return data
