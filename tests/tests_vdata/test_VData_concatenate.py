# coding: utf-8
# Created on 10/02/2021 16:54
# Author : matteo

# ====================================================
# imports
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import vdata

from . import expr_data_complex, obs_index_data


# ====================================================
# code
@pytest.fixture
def merged_vdata() -> vdata.VData:
    timepoints = pd.DataFrame({"value": ["0h", "5h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
    obs = vdata.TemporalDataFrame(
        {"data": np.random.randint(0, 20, 6), "data_bis": np.random.randint(0, 20, 6)},
        timepoints=["0h", "0h", "0h", "0h", "5h", "5h"],
        index=obs_index_data,
    )
    uns = {"colors": ["blue", "red", "yellow"], "date": "25/01/2021"}

    v1 = vdata.VData(expr_data_complex, timepoints=timepoints, obs=obs, var=var, uns=uns, name="1")

    expr_data_complex_modif = {key: TDF * -1 for key, TDF in expr_data_complex.items()}
    obs = vdata.TemporalDataFrame(
        {"data": np.random.randint(0, 20, 6), "data_bis": np.random.randint(0, 20, 6)},
        timepoints=["0h", "0h", "0h", "0h", "5h", "5h"],
        index=obs_index_data,
    )
    uns = {"colors": ["blue", "red", "pink"], "date": "24/01/2021"}

    v2 = vdata.VData(expr_data_complex_modif, timepoints=timepoints, obs=obs, var=var, uns=uns, name="2")

    v2.set_obs_index([f"C_{i}" for i in range(6, 12)])

    return vdata.concatenate((v1, v2))


def test_concatenated_VData_has_correct_shape(merged_vdata: vdata.VData) -> None:
    assert (
        repr(merged_vdata) == "VData 'No_Name' ([8, 4] obs x 3 vars over 2 time points).\n"
        "\tlayers: 'spliced', 'unspliced'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'\n"
        "\tuns: 'colors', 'date'"
    )


def test_concatenated_VData_has_correct_obs_index(merged_vdata: vdata.VData) -> None:
    assert np.all(
        merged_vdata.obs.index == ["C_0", "C_1", "C_2", "C_3", "C_6", "C_7", "C_8", "C_9", "C_4", "C_5", "C_10", "C_11"]
    )


def test_concatented_VData_has_correct_layer_index(merged_vdata: vdata.VData) -> None:
    assert np.all(
        merged_vdata.layers["spliced"].index
        == ["C_0", "C_1", "C_2", "C_3", "C_6", "C_7", "C_8", "C_9", "C_4", "C_5", "C_10", "C_11"]
    )


def test_VData_concatenate_mean() -> None:
    v3 = vdata.read(Path(__file__).parent.parent / "ref" / "vdata.vd", vdata.H5Mode.READ)
    v4 = v3.copy()

    vm3 = v3.mean(axis=0)
    vm4 = v4.mean(axis=0)

    vm4.set_obs_index(vdata.Index(["mean_2"], repeats=3))

    v_merged = vdata.concatenate((vm3, vm4))

    v_repr = """VData 'No_Name' ([2, 2, 2] obs x 3 vars over 3 time points).
	layers: 'data'
	timepoints: 'value'"""

    assert repr(v_merged) == v_repr

    v3.close()
