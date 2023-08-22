# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
import pytest

import vdata

from . import expr_data_complex, expr_data_simple, obs_index_data

# ====================================================
# code
_timepoints_data = {"value": ["0h", "5h"]}
_timepoints_data_incorrect_format = {"no_column_value": ["0h", "5h"]}
_timepoints_data_simple = {"value": ["0h"]}

_var_data = {"gene_name": ["gene 1", "gene 2", "gene 3"]}
_var_index_data = ["g1", "g2", "g3"]

_obs_data = {"data": np.random.randint(0, 20, 6), "data_bis": np.random.randint(0, 20, 6)}


# data is None
#   obs is None
#       var is None
#           time-points is None
def test_VData_creation_all_None() -> None:
    v = vdata.VData(data=None, obs=None, var=None, timepoints=None, name="1")
    assert repr(v) == "Empty VData '1' (0 obs x 0 vars over 0 time points)."


#           time-points is a pd.DataFrame
def test_VData_creation_tp_as_df() -> None:
    v = vdata.VData(data=None, obs=None, var=None, timepoints=pd.DataFrame(_timepoints_data), name="2")
    assert repr(v) == "Empty VData '2' (0 obs x 0 vars over 2 time points).\n" "\ttimepoints: 'value'"


def test_VData_creation_tp_as_invalid_df() -> None:
    with pytest.raises(ValueError) as exc_info:
        vdata.VData(data=None, obs=None, var=None, timepoints=pd.DataFrame(_timepoints_data_incorrect_format), name="3")

    assert str(exc_info.value) == "'time points' must have at least a column 'value' to store time points value."


#           time-points is invalid
def test_VData_creation_tp_is_invalid() -> None:
    with pytest.raises(TypeError) as exc_info:
        vdata.VData(data=None, obs=None, var=None, timepoints=0, name="4")

    assert str(exc_info.value) == "'time points' must be a DataFrame, got 'int'."


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_var_is_df_tp_is_None() -> None:
    v = vdata.VData(data=None, obs=None, var=pd.DataFrame(_var_data, index=_var_index_data), timepoints=None, name="5")
    assert repr(v) == "Empty VData '5' (0 obs x 3 vars over 0 time points).\n" "\tvar: 'gene_name'", repr(v)


#           time-points is a pd.DataFrame
def test_VData_creation_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=None,
        obs=None,
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        name="6",
    )
    assert (
        repr(v) == "Empty VData '6' (0 obs x 3 vars over 2 time points).\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#       var is invalid
def test_VData_creation_var_is_invalid() -> None:
    with pytest.raises(TypeError) as exc_info:
        _ = vdata.VData(data=None, obs=None, var=0, timepoints=None, name="7")

    assert str(exc_info.value) == "var must be a DataFrame."


#   obs is a pd.DataFrame
#       var is None
#           time-points is None
def test_VData_creation_obs_is_df_var_is_None_tp_is_None() -> None:
    v = vdata.VData(data=None, obs=pd.DataFrame(_obs_data, index=obs_index_data), var=None, timepoints=None, name="8")
    assert (
        repr(v) == "Empty VData '8' (6 obs x 0 vars over 1 time point).\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_obs_is_df_var_is_None_tp_is_df() -> None:
    v = vdata.VData(
        data=None,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data),
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="9",
    )
    assert (
        repr(v) == "Empty VData '9' ([4, 2] obs x 0 vars over 2 time points).\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_obs_is_df_var_is_df_tp_is_None() -> None:
    v = vdata.VData(
        data=None,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="10",
    )
    assert (
        repr(v) == "Empty VData '10' (6 obs x 3 vars over 1 time point).\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_obs_is_df_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=None,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="11",
    )
    assert (
        repr(v) == "Empty VData '11' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#   obs is a TemporalDataFrame
#       var is None
#           time-points is None
def test_VData_creation_obs_is_TDF_var_is_None_tp_is_None() -> None:
    v = vdata.VData(
        data=None,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=None,
        timepoints=None,
        name="12",
    )
    assert (
        repr(v) == "Empty VData '12' ([4, 2] obs x 0 vars over 2 time points).\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_obs_is_TDF_var_is_None_tp_is_df() -> None:
    v = vdata.VData(
        data=None,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data),
        name="13",
    )
    assert (
        repr(v) == "Empty VData '13' ([4, 2] obs x 0 vars over 2 time points).\n"
        "\tobs: 'data', 'data_bis'\n\t"
        "timepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_obs_is_TDF_var_is_df_tp_is_None() -> None:
    v = vdata.VData(
        data=None,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="14",
    )
    assert (
        repr(v) == "Empty VData '14' ([4, 2] obs x 3 vars over 2 time points).\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_obs_is_TDF_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=None,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        name="15",
    )
    assert (
        repr(v) == "Empty VData '15' ([4, 2] obs x 3 vars over 2 time points).\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#   obs is invalid
def test_VData_creation_obs_is_invalid() -> None:
    with pytest.raises(TypeError) as exc_info:
        _ = vdata.VData(data=None, obs=0, var=None, timepoints=None, name="16")

    assert str(exc_info.value) == "'obs' must be a DataFrame or a TemporalDataFrame."


# data is a pd.DataFrame
#   obs is None
#       var is None
#           time-points is None
def test_VData_creation_data_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=None,
        var=None,
        timepoints=None,
        name="17",
    )
    assert repr(v) == "VData '17' (6 obs x 3 vars over 1 time point).\n" "\tlayers: 'data'\n" "\ttimepoints: 'value'"


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=None,
        var=None,
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="18",
    )
    assert repr(v) == "VData '18' (6 obs x 3 vars over 1 time point).\n" "\tlayers: 'data'\n" "\ttimepoints: 'value'"


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_data_is_df_var_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=None,
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="19",
    )
    assert (
        repr(v) == "VData '19' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=None,
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="20",
    )
    assert (
        repr(v) == "VData '20' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#   obs is a pd.DataFrame
#       var is None
#           time-points is None
def test_VData_creation_data_is_df_obs_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=None,
        name="21",
    )
    assert (
        repr(v) == "VData '21' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_obs_id_df_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="22",
    )
    assert (
        repr(v) == "VData '22' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_data_is_df_obs_is_df_var_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="23",
    )
    assert (
        repr(v) == "VData '23' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_obs_is_df_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="24",
    )
    assert (
        repr(v) == "VData '24' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#   obs is a TemporalDataFrame
#       var is None
#           time-points is None
def test_VData_creation_data_is_df_obs_is_TDF() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=None,
        name="25",
    )
    assert (
        repr(v) == "VData '25' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_obs_is_TDF_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="26",
    )
    assert (
        repr(v) == "VData '26' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\ttimepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_data_is_df_obs_is_TDF_var_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="27",
    )
    assert (
        repr(v) == "VData '27' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_df_obs_is_TDF_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=pd.DataFrame(expr_data_simple, index=obs_index_data, columns=_var_index_data),
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data_simple),
        name="28",
    )
    assert (
        repr(v) == "VData '28' (6 obs x 3 vars over 1 time point).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'data', 'data_bis'\n"
        "\tvar: 'gene_name'\n"
        "\ttimepoints: 'value'"
    )


# data is a dict[str, np.array]
#   obs is None
#       var is None
#           time-points is None
def test_VData_creation_data_is_dict() -> None:
    v = vdata.VData(data=expr_data_complex, obs=None, var=None, timepoints=None, name="29")
    assert (
        repr(v) == "VData '29' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_tp_is_df() -> None:
    v = vdata.VData(data=expr_data_complex, obs=None, var=None, timepoints=pd.DataFrame(_timepoints_data), name="30")
    assert (
        repr(v) == "VData '30' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "timepoints: 'value'"
    )


#       var is a pd.DataFrame
#           timepoints is None
def test_VData_creation_data_is_dict_var_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex, obs=None, var=pd.DataFrame(_var_data, index=_var_index_data), timepoints=None, name="31"
    )
    assert (
        repr(v) == "VData '31' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=None,
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        name="32",
    )
    assert (
        repr(v) == "VData '32' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#   obs is a pd.DataFrame
#       var is None
#           time-points is None
def test_VData_creation_data_is_dict_obs_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=None,
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="33",
    )
    assert (
        repr(v) == "VData '33' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_obs_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data),
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="34",
    )
    assert (
        repr(v) == "VData '34' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "timepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_data_is_dict_obs_is_df_var_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="35",
    )
    assert (
        repr(v) == "VData '35' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_obs_is_df_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=pd.DataFrame(_obs_data, index=obs_index_data),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        timepoints_list=["0h", "0h", "0h", "0h", "5h", "5h"],
        name="36",
    )
    assert (
        repr(v) == "VData '36' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    ), repr(v)


#   obs is a TemporalDataFrame
#       var is None
#           time-points is None
def test_VData_creation_data_is_dict_obs_is_TDF() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=None,
        timepoints=None,
        name="37",
    )
    assert (
        repr(v) == "VData '37' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_obs_is_TDF_tp_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=None,
        timepoints=pd.DataFrame(_timepoints_data),
        name="38",
    )
    assert (
        repr(v) == "VData '38' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "timepoints: 'value'"
    )


#       var is a pd.DataFrame
#           time-points is None
def test_VData_creation_data_is_dict_obs_is_TDF_var_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=None,
        name="39",
    )
    assert (
        repr(v) == "VData '39' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


#           time-points is a pd.DataFrame
def test_VData_creation_data_is_dict_obs_is_TDF_var_is_df_tp_is_df() -> None:
    v = vdata.VData(
        data=expr_data_complex,
        obs=vdata.TemporalDataFrame(_obs_data, index=obs_index_data, timepoints=["0h", "0h", "0h", "0h", "5h", "5h"]),
        var=pd.DataFrame(_var_data, index=_var_index_data),
        timepoints=pd.DataFrame(_timepoints_data),
        name="40",
    )
    assert (
        repr(v) == "VData '40' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'"
    )


# data is invalid
def test_VData_creation_data_is_invalid() -> None:
    with pytest.raises(TypeError) as exc_info:
        vdata.VData(data=0, obs=None, var=None, timepoints=None, name="41")

    assert (
        str(exc_info.value) == "Type '<class 'int'>' is not allowed for 'data' parameter, should be a dict,"
        "a pandas DataFrame, a TemporalDataFrame or an AnnData object."
    )


def test_VData_creation_with_uns() -> None:
    timepoints = pd.DataFrame({"value": ["0h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = vdata.TemporalDataFrame(
        {"data": np.random.randint(0, 20, 6), "data_bis": np.random.randint(0, 20, 6)},
        timepoints=["0h", "0h", "0h", "0h", "0h", "0h"],
    )
    uns = {"colors": ["blue", "red", "yellow"], "date": "25/01/2021"}

    data = pd.DataFrame(expr_data_simple)

    v = vdata.VData(data, timepoints=timepoints, obs=obs, var=var, uns=uns, name="46")
    assert (
        repr(v) == "VData '46' (6 obs x 3 vars over 1 time point).\n\t"
        "layers: 'data'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'\n\t"
        "uns: 'colors', 'date'"
    )


def test_VData_creation_full() -> None:
    timepoints = pd.DataFrame({"value": ["0h", "5h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
    obs = vdata.TemporalDataFrame(
        {"data": np.random.randint(0, 20, 6), "data_bis": np.random.randint(0, 20, 6)},
        timepoints=["0h", "0h", "0h", "0h", "5h", "5h"],
        index=obs_index_data,
        name="obs",
    )
    uns = {"colors": ["blue", "red", "yellow"], "date": "25/01/2021"}

    obsm = {
        "umap": pd.DataFrame({"X1": [4, 5, 6, 7, 8, 9], "X2": [1, 2, 3, 9, 8, 7]}, index=obs_index_data),
        "pca": pd.DataFrame({"X1": [-4, -5, -6, -7, -8, -9], "X2": [-1, -2, -3, -9, -8, -7]}, index=obs_index_data),
    }
    obsp = {
        "pair": np.array(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 2, 2],
            ]
        )
    }
    varm = {"test": pd.DataFrame({"col": [7, 8, 9]}, index=["g1", "g2", "g3"])}
    varp = {"test2": pd.DataFrame({"g1": [0, 0, 1], "g2": [0, 1, 0], "g3": [1, 0, 0]}, index=["g1", "g2", "g3"])}

    v = vdata.VData(
        expr_data_complex,
        timepoints=timepoints,
        obs=obs,
        var=var,
        uns=uns,
        obsm=obsm,
        obsp=obsp,
        varm=varm,
        varp=varp,
        name="47",
    )

    assert (
        repr(v) == "VData '47' ([4, 2] obs x 3 vars over 2 time points).\n\t"
        "layers: 'spliced', 'unspliced'\n\t"
        "obs: 'data', 'data_bis'\n\t"
        "var: 'gene_name'\n\t"
        "timepoints: 'value'\n\t"
        "obsm: 'umap', 'pca'\n\t"
        "varm: 'test'\n\t"
        "obsp: 'pair'\n\t"
        "varp: 'test2'\n\t"
        "uns: 'colors', 'date'"
    ), repr(v)

    v.set_obs_index([f"C_{i}" for i in range(6, 12)])

    assert (
        repr(v.obsp["pair"]) == "      C_6  C_7  C_8  C_9  C_10  C_11\n"
        "C_6     1    1    1    1     0     0\n"
        "C_7     1    1    1    1     0     0\n"
        "C_8     1    1    1    1     0     0\n"
        "C_9     1    1    1    1     0     0\n"
        "C_10    0    0    0    0     2     2\n"
        "[RAM]\n"
        "[6 rows x 6 columns]"
    )


# def test_VData_creation_from_dict() -> None:
#     v = vdata.read_from_dict(data, name='1')

#     assert repr(v) == "VData '1' [7, 3, 10] x 4 over 3 time points.\n\t" \
#                       "layers: 'RNA', 'Protein'\n\t" \
#                       "timepoints: 'value'", repr(v)

#     timepoints = pd.DataFrame({"value": ['0h', '5h', '10h']})
#     var = pd.DataFrame({"gene_name": ["g1", "g2", "g3", "g4"]}, index=["g1", "g2", "g3", "g4"])
#     obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 20),
#                                    'data_bis': np.random.randint(0, 20, 20)},
#                                   timepoints=["0h" for _ in range(7)] + ["5h" for _ in range(3)] + ["10h" for _ in
#                                                                                                    range(10)],
#                                   index=[f"C_{i}" for i in range(20)], name='obs')

#     v = vdata.read_from_dict(data, obs=obs, var=var, timepoints=timepoints, name='2')
#     assert repr(v) == "VData '2' [7, 3, 10] x 4 over 3 time points.\n\t" \
#                       "layers: 'RNA', 'Protein'\n\t" \
#                       "obs: 'data', 'data_bis'\n\t" \
#                       "var: 'gene_name'\n\t" \
#                       "timepoints: 'value'"


def test_VData_creation_repeating_index() -> None:
    obs = vdata.TemporalDataFrame(
        data={"col1": np.arange(9)},
        timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
        index=vdata.Index(["a", "b", "c"], repeats=3),
    )

    data = vdata.TemporalDataFrame(
        data={
            "G1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "G2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "G3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        },
        timepoints=["0h", "0h", "0h", "5h", "5h", "5h", "10h", "10h", "10h"],
        index=vdata.Index(["a", "b", "c"], repeats=3),
    )

    v = vdata.VData(data, obs=obs)
    assert (
        repr(v) == "VData 'No_Name' ([3, 3, 3] obs x 3 vars over 3 time points).\n"
        "\tlayers: 'data'\n"
        "\tobs: 'col1'\n"
        "\ttimepoints: 'value'"
    )
