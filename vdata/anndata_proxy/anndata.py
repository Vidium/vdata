from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from h5dataframe import H5DataFrame
from scipy import sparse

from vdata._typing import AnyNDArrayLike, DictLike
from vdata.anndata_proxy.containers import H5DataFrameContainerProxy, TemporalDataFrameContainerProxy
from vdata.anndata_proxy.dataframe import DataFrameProxy_TDF
from vdata.data._file import NoData

if TYPE_CHECKING:
    from vdata.data import VData, VDataView


def skip_time_axis(slicer: Any) -> tuple[Any, ...]:
    if isinstance(slice, tuple):
        return (slice(None),) + slicer

    return (slice(None), slicer)


class AnnDataProxy(AnnData):  # type: ignore[misc]
    """
    Class faking to be an anndata.AnnData object but actually wrapping a VData.
    """

    __slots__ = "_vdata", "_X", "_layers", "_obs", "_obsm", "_obsp", "_var", "_varm", "_varp", "_uns"

    # region magic methods
    def __init__(self, vdata: VData | VDataView, X: str | None = None) -> None:
        """
        Args:
            vdata: a VData object to wrap.
            X: an optional layer name to use as X.
        """
        self._X = None if X is None else str(X)

        if self._X is not None and self._X not in vdata.layers:
            raise ValueError(f"Could not find layer '{self._X}' in the given VData.")

        self._init_from_vdata(vdata)

    def _init_from_vdata(self, vdata: VData | VDataView) -> None:
        self._vdata = vdata
        self._layers = TemporalDataFrameContainerProxy(vdata, name="layers")
        self._obs = DataFrameProxy_TDF(vdata.obs)
        self._obsm = TemporalDataFrameContainerProxy(vdata, name="obsm")
        self._obsp = H5DataFrameContainerProxy(vdata.obsp, name="Obsp", index=vdata.obs.index, columns=vdata.obs.index)
        self._var = vdata.var
        self._varm = H5DataFrameContainerProxy(vdata.varm, name="Varm", index=vdata.var.index)
        self._varp = H5DataFrameContainerProxy(vdata.varp, name="Varp", index=vdata.var.index, columns=vdata.var.index)
        self._uns = vdata.uns

    def __repr__(self) -> str:
        return f"AnnDataProxy from {self._vdata}"

    def __sizeof__(self, show_stratified: bool | None = None) -> int:
        del show_stratified
        raise NotImplementedError

    def __delitem__(self, index: Any) -> None:
        raise NotImplementedError

    def __getitem__(self, index: Any) -> AnnDataProxy:
        """Returns a sliced view of the object."""
        return AnnDataProxy(self._vdata[skip_time_axis(index)], X=self._X)

    def __setitem__(self, index: Any, val: int | float | npt.NDArray[Any] | sparse.spmatrix) -> None:
        raise NotImplementedError

    # endregion

    # region attributes
    @property
    def _n_obs(self) -> int:
        return self._vdata.n_obs_total

    @property
    def _n_vars(self) -> int:
        return self._vdata.n_var

    @property
    def X(self) -> AnyNDArrayLike[Any] | None:
        if self._X is None:
            return None
        return self._vdata.layers[self._X].values

    @X.setter
    def X(self, value: Any) -> None:
        raise NotImplementedError

    @X.deleter
    def X(self) -> None:
        self._X = None

    @property
    def layers(self) -> TemporalDataFrameContainerProxy:
        return self._layers

    @layers.setter
    def layers(self, value: Any) -> None:
        del value
        raise NotImplementedError

    @layers.deleter
    def layers(self) -> None:
        raise NotImplementedError

    @property
    def raw(self) -> None:
        raise NotImplementedError

    @raw.setter
    def raw(self, value: AnnData) -> None:
        raise NotImplementedError

    @raw.deleter
    def raw(self) -> None:
        raise NotImplementedError

    @property
    def obs(self) -> DataFrameProxy_TDF:
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        raise NotImplementedError

    @obs.deleter
    def obs(self) -> None:
        raise NotImplementedError

    @property
    def obs_names(self) -> pd.Index:
        """Names of observations (alias for `.obs.index`)."""
        return self.obs.index

    @obs_names.setter
    def obs_names(self, names: Sequence[str]) -> None:
        raise NotImplementedError

    @property
    def var(self) -> H5DataFrame:
        """One-dimensional annotation of variables/ features (`pd.DataFrame`)."""
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        raise NotImplementedError

    @var.deleter
    def var(self) -> None:
        raise NotImplementedError

    @property
    def var_names(self) -> pd.Index:
        """Names of variables (alias for `.var.index`)."""
        return self.var.index

    @var_names.setter
    def var_names(self, names: Sequence[str]) -> None:
        raise NotImplementedError

    @property
    def uns(self) -> DictLike[Any]:
        """Unstructured annotation (ordered dictionary)."""
        return self._uns

    @uns.setter
    def uns(self, value: DictLike[Any]) -> None:
        raise NotImplementedError

    @uns.deleter
    def uns(self) -> None:
        raise NotImplementedError

    @property
    def obsm(self) -> TemporalDataFrameContainerProxy:
        return self._obsm

    @obsm.setter
    def obsm(self, value: Any) -> None:
        raise NotImplementedError

    @obsm.deleter
    def obsm(self) -> None:
        raise NotImplementedError

    @property
    def varm(self) -> H5DataFrameContainerProxy:
        return self._varm

    @varm.setter
    def varm(self, value: Any) -> None:
        raise NotImplementedError

    @varm.deleter
    def varm(self) -> None:
        raise NotImplementedError

    @property
    def obsp(self) -> H5DataFrameContainerProxy:
        return self._obsp

    @obsp.setter
    def obsp(self, value: Any) -> None:
        raise NotImplementedError

    @obsp.deleter
    def obsp(self) -> None:
        raise NotImplementedError

    @property
    def varp(self) -> H5DataFrameContainerProxy:
        return self._varp

    @varp.setter
    def varp(self, value: Any) -> None:
        raise NotImplementedError

    @varp.deleter
    def varp(self) -> None:
        raise NotImplementedError

    @property
    def filename(self) -> Path | None:
        if self._vdata.data is NoData._:
            return None
        return Path(self._vdata.data.filename)

    @filename.setter
    def filename(self, filename: Path | None) -> None:
        raise NotImplementedError

    # endregion

    # region predicates
    @property
    def isbacked(self) -> bool:
        """`True` if object is backed on disk, `False` otherwise."""
        return self._vdata.is_backed

    @property
    def is_view(self) -> bool:
        """`True` if object is view of another AnnData object, `False` otherwise."""
        return self._vdata.is_view

    # endregion

    # region methods
    def as_vdata(self) -> VData | VDataView:
        return self._vdata

    def rename_categories(self, key: str, categories: Sequence[Any]) -> None:
        raise NotImplementedError

    def strings_to_categoricals(self, df: pd.DataFrame | None = None) -> None:
        raise NotImplementedError

    def _inplace_subset_var(self, index: Any) -> None:
        self._init_from_vdata(self._vdata[skip_time_axis((slice(None), index))])

    def _inplace_subset_obs(self, index: Any) -> None:
        self._init_from_vdata(self._vdata[skip_time_axis(index)])

    def copy(self, filename: Path | None = None) -> None:
        """Full copy, optionally on disk."""
        raise NotImplementedError

    def write_h5ad(
        self,
        filename: Path | None = None,
        compression: Literal["gzip", "lzf"] | None = None,
        compression_opts: Any = None,
        as_dense: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError

    # endregion
