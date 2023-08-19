# coding: utf-8
# Created on 11/25/20 3:41 PM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
import pytest

import vdata
from vdata import TemporalDataFrame
from vdata.timepoint import TimePointArray

# ====================================================
# code
_MASK_GENES = ["g_10", "g_2", "g_5", "g_25", "g_49"]
_MASK_CELLS = ["c_20", "c_1", "c_199", "c_100", "c_30", "c_10", "c_150", "c_151"]
_INDEX_CELLS = ["c_20", "c_1", "c_30", "c_10", "c_199", "c_100", "c_150", "c_151"]


@pytest.fixture
def sub_vdata() -> vdata.VDataView:
    genes = list(map(lambda x: "g_" + str(x), range(50)))
    cells = list(map(lambda x: "c_" + str(x), range(300)))

    v = vdata.VData(
        data={"data": pd.DataFrame(np.array(range(300 * 50)).reshape((300, 50)), index=cells, columns=genes)},
        obs=pd.DataFrame({"col1": range(300)}, index=cells),
        var=pd.DataFrame({"col1": range(50)}, index=genes),
        time_list=["0h" for _ in range(100)] + ["1h" for _ in range(100)] + ["2h" for _ in range(100)],
    )

    v.obsm["test_obsm"] = vdata.TemporalDataFrame(
        pd.DataFrame(np.array(range(v.n_obs_total * 2)).reshape((v.n_obs_total, 2)), columns=["X1", "X2"]),
        index=v.obs.index,
        time_list=v.obs.timepoints_column,
    )

    v.obsp["test_obsp"] = pd.DataFrame(
        np.array(range(v.n_obs_total * v.n_obs_total)).reshape((v.n_obs_total, v.n_obs_total)),
        columns=v.obs.index,
        index=v.obs.index,
    )

    v.varm["test_varm"] = pd.DataFrame(
        np.array(range(v.n_var * 2)).reshape((v.n_var, 2)), index=v.var.index, columns=["X1", "X2"]
    )

    v.varp["test_varp"] = pd.DataFrame(
        np.array(range(v.n_var * v.n_var)).reshape((v.n_var, v.n_var)), columns=v.var.index, index=v.var.index
    )

    return v[:, _MASK_CELLS, _MASK_GENES]


def test_VData_sub_setting_creation(sub_vdata: vdata.VDataView) -> None:
    assert (
        repr(sub_vdata) == "View of VData 'No_Name' ([4, 4] obs x 5 vars over 2 time points).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'col1'\n"
        "\tvar: 'col1'\n"
        "\ttimepoints: 'value'\n"
        "\tobsm: 'test_obsm'\n"
        "\tvarm: 'test_varm'\n"
        "\tobsp: 'test_obsp'\n"
        "\tvarp: 'test_varp'"
    ), repr(sub_vdata)


def test_VData_sub_setting_correct_layers_index(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.layers["data"].index == _INDEX_CELLS), sub_vdata.layers["data"].index


def test_VData_sub_setting_correct_layers_columns(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.layers["data"].columns == _MASK_GENES), sub_vdata.layers["data"].columns


def test_VData_sub_setting_correct_layers_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(
        sub_vdata.layers["data"].values_num.flatten(),
        np.array([int(c[2:]) * 50 + int(g[2:]) for c in _INDEX_CELLS for g in _MASK_GENES]),
    )


def test_VData_sub_setting_correct_obs_index(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.obs.index == _INDEX_CELLS), sub_vdata.obs.index


def test_VData_sub_setting_correct_obs_columns(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.obs.columns == ["col1"]), sub_vdata.obs.columns


def test_VData_sub_setting_correct_obs_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(sub_vdata.obs.values_num.flatten(), np.array([int(c[2:]) for c in _INDEX_CELLS]))


def test_VData_sub_setting_correct_var_index(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.var.index.equals(pd.Index(_MASK_GENES)), sub_vdata.var.index


def test_VData_sub_setting_correct_var_columns(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.var.columns == ["col1"]), sub_vdata.var.columns


def test_VData_sub_setting_correct_var_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(sub_vdata.var.values.flatten(), np.array([int(g[2:]) for g in _MASK_GENES]))


def test_VData_sub_setting_correct_obsm_index(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.obsm["test_obsm"].index == _INDEX_CELLS), sub_vdata.obsm["test_obsm"].index


def test_VData_sub_setting_correct_obms_columns(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.obsm["test_obsm"].columns == ["X1", "X2"]), sub_vdata.obsm["test_obsm"].columns


def test_VData_sub_setting_correct_obsm_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(
        sub_vdata.obsm["test_obsm"].values_num.flatten(),
        np.array([int(c[2:]) * 2 + col for c in _INDEX_CELLS for col in (0, 1)]),
    )


def test_VData_sub_setting_correct_obsp_index(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.obsp["test_obsp"].index.equals(pd.Index(_INDEX_CELLS)), sub_vdata.obsp["test_obsp"].index


def test_VData_sub_setting_correct_obsp_columns(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.obsp["test_obsp"].columns.equals(pd.Index(_INDEX_CELLS)), sub_vdata.obsp["test_obsp"].columns


def test_VData_sub_setting_correct_obsp_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(
        sub_vdata.obsp["test_obsp"].values.flatten(),
        np.array([int(c[2:]) * 300 + int(c2[2:]) for c in _INDEX_CELLS for c2 in _INDEX_CELLS]),
    )


def test_VData_sub_setting_correct_varm_index(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.varm["test_varm"].index.equals(pd.Index(_MASK_GENES)), sub_vdata.varm["test_varm"].index


def test_VData_sub_setting_correct_varm_columns(sub_vdata: vdata.VDataView) -> None:
    assert np.all(sub_vdata.varm["test_varm"].columns == ["X1", "X2"]), sub_vdata.varm["test_varm"].columns


def test_VData_sub_setting_correct_varm_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(
        sub_vdata.varm["test_varm"].values.flatten(),
        np.array([int(g[2:]) * 2 + col for g in _MASK_GENES for col in (0, 1)]),
    )


def test_VData_sub_setting_correct_varp_index(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.varp["test_varp"].index.equals(pd.Index(_MASK_GENES)), sub_vdata.varp["test_varp"].index


def test_VData_sub_setting_correct_varp_columns(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.varp["test_varp"].columns.equals(pd.Index(_MASK_GENES)), sub_vdata.varp["test_varp"].columns


def test_VData_sub_setting_correct_varp_values(sub_vdata: vdata.VDataView) -> None:
    assert np.array_equal(
        sub_vdata.varp["test_varp"].values.flatten(),
        np.array([int(g[2:]) * 50 + int(g2[2:]) for g in _MASK_GENES for g2 in _MASK_GENES]),
    )


def test_VData_sub_setting_correct_shape(sub_vdata: vdata.VDataView) -> None:
    assert sub_vdata.n_obs_total == len(_INDEX_CELLS), sub_vdata.n_obs_total
    assert sub_vdata.n_var == len(_MASK_GENES), sub_vdata.n_var

    assert sub_vdata.layers.shape == (1, 2, [4, 4], 5)
    assert sub_vdata.layers.shape == (1, 2, sub_vdata.n_obs, sub_vdata.n_var)


def test_vdata_subset_on_timepoints_should_set_new_values_in_obs(VData: vdata.VData) -> None:
    v = VData[["1h", "2h"]]
    v.copy()

    subset = VData[["1h", "2h"]].copy()

    new_data = TemporalDataFrame(
        {"col1": np.arange(subset.n_obs_total)},
        index=subset.obs.index,
        time_list=subset.obs.timepoints_column,
        name="new_data",
    )

    subset.obsm["new_data"] = new_data

    assert "new_data" in subset.obsm.keys()


def test_vdata_subset_timepoints_range(VData: vdata.VData) -> None:
    v = VData[1:3]

    assert np.array_equal(v.timepoints.value, TimePointArray([1, 2], unit="h"))
