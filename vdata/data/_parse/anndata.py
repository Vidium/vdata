from typing import Any, Collection

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from h5dataframe import H5DataFrame
from scipy.sparse import spmatrix

from vdata.data._parse.data import ParsingDataIn, ParsingDataOut
from vdata.data._parse.time import check_time_match
from vdata.data._parse.utils import log_timepoints
from vdata.IO.logger import generalLogger
from vdata.tdf import TemporalDataFrame, TemporalDataFrameView
from vdata.utils import deep_dict_convert


def _no_dense_data(_data: npt.NDArray[Any] | spmatrix) -> npt.NDArray[Any]:
    """
    Convert sparse matrices to dense.
    """
    if isinstance(_data, spmatrix):
        raise NotImplementedError

    return _data


def array_isin(array: npt.NDArray[Any], list_arrays: npt.NDArray[Any] | Collection[npt.NDArray[Any]]) -> bool:
    """
    Whether a given array is in a collection of arrays.
    :param array: an array.
    :param list_arrays: a collection of arrays.
    :return: whether the array is in the collection of arrays.
    """
    for target_array in list_arrays:
        if np.array_equal(array, target_array):
            return True

    return False


def parse_AnnData(adata: AnnData, data: ParsingDataIn) -> ParsingDataOut:
    generalLogger.debug("  VData creation from an AnnData.")

    # import and cast obs to a TemporalDataFrame
    obs = TemporalDataFrame(
        adata.obs,
        timepoints=data.timepoints_list,
        time_col_name=data.time_col_name,
        name="obs",
        lock=(True, False),
    )
    reordering_index = obs.index

    check_time_match(data)
    log_timepoints(data.timepoints)

    if array_isin(adata.X, adata.layers.values()):
        layers = dict(
            (
                key,
                TemporalDataFrame(
                    pd.DataFrame(arr, index=adata.obs.index, columns=adata.var.index).reindex(
                        np.array(reordering_index)
                    ),
                    timepoints=obs.timepoints_column,
                    name=key,
                ),
            )
            for key, arr in adata.layers.items()
        )

    else:
        layers = dict(
            {
                "data": TemporalDataFrame(
                    pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index).reindex(
                        np.array(reordering_index)
                    ),
                    timepoints=obs.timepoints_column,
                    name="adata",
                )
            },
            **dict(
                (
                    key,
                    TemporalDataFrame(
                        pd.DataFrame(arr, index=adata.obs.index, columns=adata.var.index).reindex(
                            np.array(reordering_index)
                        ),
                        timepoints=obs.timepoints_column,
                        name=key,
                    ),
                )
                for key, arr in adata.layers.items()
            )
        )

    # import other arrays
    obsm: dict[str, TemporalDataFrame | TemporalDataFrameView] = {
        TDF_name: TemporalDataFrame(
            pd.DataFrame(_no_dense_data(TDF_data)), timepoints=obs.timepoints_column, index=obs.index, name=TDF_name
        )
        for TDF_name, TDF_data in adata.obsm.items()
    }
    obsp = {
        VDF_name: H5DataFrame(_no_dense_data(VDF_data), index=np.array(obs.index), columns=np.array(obs.index))
        for VDF_name, VDF_data in adata.obsp.items()
    }
    var = H5DataFrame(adata.var)
    varm = {
        VDF_name: H5DataFrame(_no_dense_data(VDF_data), index=var.index) for VDF_name, VDF_data in adata.varm.items()
    }
    varp = {
        VDF_name: H5DataFrame(_no_dense_data(VDF_data), index=var.index, columns=var.index)
        for VDF_name, VDF_data in adata.varp.items()
    }
    uns = deep_dict_convert(adata.uns)
    timepoints = data.timepoints if isinstance(data.timepoints, H5DataFrame) else H5DataFrame(data.timepoints)

    return ParsingDataOut(layers, obs, obsm, obsp, var, varm, varp, timepoints, uns)
