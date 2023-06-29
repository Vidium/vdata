from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from anndata import AnnData

from vdata.anndata_proxy.dataframe import DataFrameproxy
from vdata.anndata_proxy.layers import LayersProxy

if TYPE_CHECKING:
    from vdata.data import VData, VDataView


class AnnDataProxy(AnnData):  # type: ignore[misc]
    """
    Class faking to be an anndata.AnnData object but actually wrapping a VData.
    """

    __slots__ = "_vdata", "_X", "_layers", "_obs"

    # region magic methods
    def __init__(self, vdata: VData | VDataView, X: str | None = None) -> None:
        """
        Args:
            vdata: a VData object to wrap.
            X: an optional layer name to use as X.
        """
        self._vdata = vdata
        self._X = None if X is None else str(X)
        self._layers = LayersProxy(vdata.layers)
        self._obs: DataFrameproxy = DataFrameproxy(vdata.obs)

        if self._X not in vdata.layers:
            raise ValueError(f"Could not find layer '{self._X}' in the given VData.")

    def __repr__(self) -> str:
        return f"AnnDataProxy from {self._vdata}."

    def __sizeof__(self, show_stratified: bool | None = None) -> int:
        del show_stratified
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
    def X(self) -> npt.NDArray[Any] | None:
        if self._X is None:
            return None
        return self._vdata.layers[self._X].values

    @X.setter
    def X(self, value: Any) -> None:
        raise NotImplementedError

    @X.deleter
    def X(self) -> None:
        self._X = None

    # endregion

    # region methods
    def as_vdata(self) -> VData | VDataView:
        return self._vdata

    # endregion
