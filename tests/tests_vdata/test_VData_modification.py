# coding: utf-8
# Created on 04/05/2022 17:22
# Author : matteo

# ====================================================
# imports
from pathlib import Path

import numpy as np
import pandas as pd

import vdata

# ====================================================
# code
REF_DIR = Path(__file__).parent.parent / "ref"


def test_VData_modification() -> None:
    v1 = vdata.VData.read(REF_DIR / "vdata.vd", mode="r+")

    # set once
    v1.obsm["X"] = vdata.TemporalDataFrame(
        data=pd.DataFrame({"col1": range(v1.n_obs_total)}), timepoints=v1.obs.timepoints_column, index=v1.obs.index
    )

    assert "X" in v1.obsm.keys()

    # set a second time
    v1.obsm["X"] = vdata.TemporalDataFrame(
        data=pd.DataFrame({"col1": 2 * np.arange(v1.n_obs_total)}),
        timepoints=v1.obs.timepoints_column,
        index=v1.obs.index,
    )

    assert "X" in v1.obsm.keys()
    del v1.obsm["X"]
    assert "X" not in v1.obsm.keys()

    v1.close()


def test_VData_set_index(VData: vdata.VData) -> None:
    # v1 = vdata.VData.read(REF_DIR / "vdata.vd", mode='r+')

    # not repeating ==> not repeating
    VData.set_obs_index(values=range(VData.n_obs_total))

    assert np.array_equal(VData.obs.index, range(VData.n_obs_total))
    assert np.array_equal(VData.layers["data"].index, range(VData.n_obs_total))

    # v1.close()


def test_VData_set_index_repeating(VData: vdata.VData) -> None:
    # not repeating ==> repeating
    new_index = vdata.Index(range(100), repeats=VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)

    # repeating ==> repeating
    new_index = vdata.Index(range(0, 200, 2), repeats=VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)

    # repeating ==> not repeating
    new_index = np.arange(100 * VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)
