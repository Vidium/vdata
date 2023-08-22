from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import ch5mpy as ch
import numpy as np
import pandas as pd
from anndata import AnnData
from h5dataframe import H5DataFrame
from scipy.sparse import spmatrix

import vdata.timepoint as tp
from vdata._typing import NDArray_IFS
from vdata.array_view import NDArrayView
from vdata.data._parse.objects import get_obs_index, get_var_index
from vdata.data._parse.time import parse_timepoints, parse_timepoints_list
from vdata.IO.errors import IncoherenceError
from vdata.IO.logger import generalLogger
from vdata.names import NO_NAME
from vdata.tdf import TemporalDataFrame, TemporalDataFrameBase, TemporalDataFrameView
from vdata.utils import first_in


def at_least_empty_dict(d: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    return {} if d is None else d


def _get_time_list(
    time_list: tp.TimePointArray | NDArrayView[tp.TimePoint] | None,
    data: Any,
    time_col_name: str | None,
) -> tp.TimePointArray | NDArrayView[tp.TimePoint] | None:
    if time_list is not None:
        return time_list

    if isinstance(data, TemporalDataFrameBase):
        return data.timepoints_column

    if isinstance(data, dict):
        df = first_in(data)

        if isinstance(df, TemporalDataFrameBase):
            return df.timepoints_column

        elif isinstance(df, (pd.DataFrame, H5DataFrame)) and time_col_name is not None:
            return tp.as_timepointarray(df[time_col_name])

    return None


def _valid_obs(
    data: pd.DataFrame
    | H5DataFrame
    | TemporalDataFrameBase
    | Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
    | None,
    obs: pd.DataFrame | H5DataFrame | TemporalDataFrameBase | None,
    time_list: tp.TimePointArray | NDArrayView[tp.TimePoint] | None,
    time_col_name: str | None,
) -> TemporalDataFrameBase:
    if obs is None:
        generalLogger.debug("Default empty TemporalDataFrame for obs.")

        _obs_index = get_obs_index(data, obs)
        _time_list = _get_time_list(time_list, data, time_col_name)

        _obs = TemporalDataFrame(
            timepoints=_time_list,
            index=_obs_index,
            name="obs",
            lock=(True, False),
        )
        _obs.lock_indices()
        return _obs

    generalLogger.debug(lambda: f"    2. \u2713 'obs' is a {type(obs).__name__}.")

    if isinstance(obs, (pd.DataFrame, H5DataFrame)):
        _obs = TemporalDataFrame(obs, timepoints=time_list, time_col_name=time_col_name, name="obs", index=obs.index)
        _obs.lock_indices()
        return _obs

    elif isinstance(obs, TemporalDataFrame):
        obs.lock_indices()
        obs.unlock_columns()

        if obs.name != "obs":
            obs.name = "obs" if obs.name == NO_NAME else f"{obs.name}_obs"

        return obs

    raise TypeError("'obs' must be a DataFrame or a TemporalDataFrame.")


def _valid_var(
    data: pd.DataFrame
    | H5DataFrame
    | TemporalDataFrameBase
    | Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
    | None,
    var: pd.DataFrame | H5DataFrame | None,
    time_col_name: str | None,
) -> H5DataFrame:
    if var is None:
        generalLogger.debug("Default empty DataFrame for vars.")
        _index = get_var_index(data, var)

        if _index is not None and time_col_name is not None:
            ix = np.where(_index == time_col_name)[0][0]
            _index = np.delete(_index, ix)

        return H5DataFrame(pd.DataFrame(index=_index))

    if isinstance(var, (pd.DataFrame, H5DataFrame)):
        generalLogger.debug(lambda: f"    5. \u2713 'var' is a {type(var).__name__}.")
        return H5DataFrame(var)

    raise TypeError("var must be a DataFrame.")


@dataclass
class ParsingDataIn:
    data: pd.DataFrame | H5DataFrame | TemporalDataFrameBase | Mapping[
        str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase
    ] | None
    obs: pd.DataFrame | H5DataFrame | TemporalDataFrameBase
    obsm: Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
    obsp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS]
    var: pd.DataFrame | H5DataFrame
    varm: Mapping[str, pd.DataFrame | H5DataFrame]
    varp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS]
    timepoints: pd.DataFrame | H5DataFrame
    time_col_name: str | None
    timepoints_list: tp.TimePointArray | NDArrayView[tp.TimePoint] | None
    uns: Mapping[str, Any]
    layers: dict[str, TemporalDataFrame | TemporalDataFrameView] = field(init=False)

    def __post_init__(self) -> None:
        self.obsm = at_least_empty_dict(self.obsm)
        self.obsp = at_least_empty_dict(self.obsp)
        self.varm = at_least_empty_dict(self.varm)
        self.varp = at_least_empty_dict(self.varp)
        self.uns = at_least_empty_dict(self.uns)

        self.layers = {}

    @classmethod
    def from_objects(
        cls,
        data: pd.DataFrame
        | TemporalDataFrameBase
        | Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase]
        | None,
        obs: pd.DataFrame | H5DataFrame | TemporalDataFrameBase | None,
        obsm: Mapping[str, pd.DataFrame | H5DataFrame | TemporalDataFrameBase] | None,
        obsp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS] | None,
        var: pd.DataFrame | H5DataFrame | None,
        varm: Mapping[str, pd.DataFrame | H5DataFrame] | None,
        varp: Mapping[str, pd.DataFrame | H5DataFrame | NDArray_IFS] | None,
        timepoints: pd.DataFrame | H5DataFrame | None,
        time_col_name: str | None,
        timepoints_list: Sequence[str | tp.TimePoint] | tp.TimePointArray | None,
        uns: dict[str, Any] | ch.H5Dict[Any] | None,
    ) -> ParsingDataIn:
        _timepoints_list = parse_timepoints_list(timepoints_list, time_col_name, obs)

        return ParsingDataIn(
            data,
            _valid_obs(data, obs, _timepoints_list, time_col_name),
            at_least_empty_dict(obsm),
            at_least_empty_dict(obsp),
            _valid_var(data, var, time_col_name),
            at_least_empty_dict(varm),
            at_least_empty_dict(varp),
            parse_timepoints(timepoints),
            time_col_name,
            _timepoints_list,
            at_least_empty_dict(uns),
        )

    @classmethod
    def from_anndata(
        cls,
        adata: AnnData,
        obs: Any,
        obsm: Any,
        obsp: Any,
        var: Any,
        varm: Any,
        varp: Any,
        timepoints: Any,
        time_col_name: Any,
        timepoints_list: Any,
        uns: Any,
    ) -> ParsingDataIn:
        if isinstance(adata.X, spmatrix):
            # adata.X = adata.X.toarray()
            raise NotImplementedError(f"'X' is a {type(adata.X).__name__}, sparse matrices are not handled yet.")

        for layer_name in adata.layers:
            if isinstance(adata.layers[layer_name], spmatrix):
                # adata.layers[layer_name] = adata.layers[layer_name].toarray()
                raise NotImplementedError(
                    f"layer {layer_name} is a {type(adata.layers[layer_name]).__name__}, "
                    f"sparse matrices are not handled yet."
                )

        # if an AnnData is being imported, obs, obsm, obsp, var, varm, varp and uns should be None because
        # they will be set from the AnnData
        for attr_name, attr in (
            ("obs", obs),
            ("obsm", obsm),
            ("obsp", obsp),
            ("var", var),
            ("varm", varm),
            ("varp", varp),
            ("uns", uns),
        ):
            if attr is not None:
                raise ValueError(f"'{attr_name}' should be None for VData creation from an AnnData.")

        return ParsingDataIn(
            data=None,
            obs=adata.obs,
            obsm=adata.obsm,
            obsp=adata.obsp,
            var=adata.var,
            varm=adata.varm,
            varp=adata.varp,
            timepoints=parse_timepoints(timepoints),
            time_col_name=time_col_name,
            timepoints_list=parse_timepoints_list(timepoints_list, time_col_name, adata.obs),
            uns=adata.uns,
        )


@dataclass
class ParsingDataOut:
    """Output class of the parsing logic. It checks for incoherence in the arrays."""

    layers: dict[str, TemporalDataFrame | TemporalDataFrameView]
    obs: TemporalDataFrameBase
    obsm: dict[str, TemporalDataFrame | TemporalDataFrameView]
    obsp: dict[str, H5DataFrame]
    var: H5DataFrame
    varm: dict[str, H5DataFrame]
    varp: dict[str, H5DataFrame]
    timepoints: H5DataFrame
    uns: dict[str, Any]

    def __post_init__(self) -> None:
        # get shape once for performance
        n_timepoints, n_obs, n_var = len(self.timepoints), self.obs.shape[1], len(self.var)

        # check coherence with number of time points in VData
        for attr in ("layers", "obsm"):
            dataset = getattr(self, attr)
            if len(dataset) and first_in(dataset).shape[0] != n_timepoints:
                raise IncoherenceError(
                    f"{attr} has {first_in(dataset).shape[0]} time point"
                    f"{'' if first_in(dataset).shape[0] == 1 else 's'} but {n_timepoints}"
                    f" {'was' if n_timepoints == 1 else 'were'} given."
                )

        generalLogger.debug("Time points were coherent across arrays.")

        # check coherence between layers, obs, var and time points
        for layer_name, layer in self.layers.items():
            if layer.shape[0] != n_timepoints:
                raise IncoherenceError(
                    f"layer '{layer_name}' has incoherent number of time points "
                    f"{layer.shape[0]}, should be {n_timepoints}."
                )

            elif layer.shape[1] != n_obs:
                for tp_i, timepoint in enumerate(layer.timepoints):
                    if layer.n_index_at(timepoint) != n_obs[tp_i]:
                        raise IncoherenceError(
                            f"layer '{layer_name}' has incoherent number of observations "
                            f"{layer.n_index_at(timepoint)}, should be {n_obs[tp_i]}."
                        )

            elif layer.shape[2] != n_var:
                raise IncoherenceError(
                    f"layer '{layer_name}' has incoherent number of variables " f"{layer.shape[2]}, should be {n_var}."
                )

        # check coherence between obs, obsm and obsp shapes
        if len(self.obsm) and first_in(self.obsm).shape[1] != n_obs:
            raise IncoherenceError(
                f"'obs' and 'obsm' have different lengths ({n_obs} vs " f"{first_in(self.obsm).shape[1]})"
            )

        if len(self.obsp) and first_in(self.obsp).shape[1] != self.obs.n_index:
            raise IncoherenceError(
                f"'obs' and 'obsp' have different lengths ({n_obs} vs " f"{first_in(self.obsp).shape[1]})"
            )

        # check coherence between var, varm, varp shapes
        for attr in ("varm", "varp"):
            dataset = getattr(self, attr)
            if len(dataset) and first_in(dataset).shape[0] != n_var:
                raise IncoherenceError(
                    f"'var' and 'varm' have different lengths ({n_var} vs " f"{first_in(dataset).shape[0]})"
                )

    @classmethod
    def from_h5(cls, data: ch.H5Dict[Any]) -> ParsingDataOut:
        with ch.options(error_mode="raise"):
            _timepoints = data["timepoints"]
            _timepoints.value = _timepoints.value.map(lambda x: tp.TimePoint(x))

            return ParsingDataOut(
                layers=data["layers"],
                obs=data["obs"],
                obsm=data["obsm"],
                obsp=data["obsp"],
                var=data["var"],
                varm=data["varm"],
                varp=data["varp"],
                timepoints=_timepoints,
                uns=data["uns"],
            )
