# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import pytest
import numpy as np
import pandas as pd

import vdata
from . import obs_index_data, expr_data_simple, expr_data_complex, data


# ====================================================
# code
def test_VData_creation():
    timepoints_data = {"value": ["0h", "5h"]}
    timepoints_data_incorrect_format = {"no_column_value": ["0h", "5h"]}
    timepoints_data_simple = {"value": ["0h"]}

    var_data = {"gene_name": ["gene 1", "gene 2", "gene 3"]}
    var_index_data = ["g1", "g2", "g3"]

    obs_data = {'data': np.random.randint(0, 20, 6),
                'data_bis': np.random.randint(0, 20, 6)}

    # data is None
    #   obs is None
    #       var is None
    #           time-points is None
    v = vdata.VData(data=None, obs=None, var=None, timepoints=None, name=1)
    assert repr(v) == "Empty VData '1' (0 obs x 0 vars over 0 time points).\n" \
                      "\tlayers: 'data'", repr(v)

    #           time-points is a pd.DataFrame
    timepoints = pd.DataFrame(timepoints_data)

    # TODO : fix those tests
    # v = vdata.VData(data=None, obs=None, var=None, timepoints=timepoints, name=2)
    # assert repr(v) == "Empty VData '2' ([0, 0] obs x 0 vars over 2 time points).\n\ttimepoints: 'value'", \
    #     repr(v)

    timepoints = pd.DataFrame(timepoints_data_incorrect_format)

    with pytest.raises(vdata.VValueError) as exc_info:
        vdata.VData(data=None, obs=None, var=None, timepoints=timepoints, name=3)

    assert exc_info.value.msg == "'time points' must have at least a column 'value' to store time points value.", \
        repr(v)

    #           time-points is invalid
    timepoints = 0

    # TODO : fix those tests
    # with pytest.raises(vdata.VValueError) as exc_info:
    #     vdata.VData(data=None, obs=None, var=None, timepoints=timepoints, name=4)
    #
    # assert exc_info.value.msg == "'time points' must be a pandas DataFrame."

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time-points is None
    v = vdata.VData(data=None, obs=None, var=var, timepoints=None, name=5)
    assert repr(v) == "Empty VData '5' (0 obs x 3 vars over 0 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tvar: 'gene_name'", repr(v)

    #           time-points is a pd.DataFrame
    timepoints = pd.DataFrame(timepoints_data)

    # TODO : fix those tests
    # v = vdata.VData(data=None, obs=None, var=var, timepoints=timepoints, name=6)
    # assert repr(v) == "Empty VData '6' ([0, 0] obs x 3 vars over 2 time points).\n\t" \
    #                   "var: 'gene_name'\n\t" \
    #                   "timepoints: 'value'", repr(v)

    #       var is invalid
    var = 0

    # TODO : fix those tests
    # with pytest.raises(vdata.VValueError) as exc_info:
    #     _ = vdata.VData(data=None, obs=None, var=var, timepoints=None, name=7)
    #
    # assert exc_info.value.msg == "var must be a pandas DataFrame."

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time-points is None
    v = vdata.VData(data=None, obs=obs, var=None, timepoints=None, name=8)
    assert repr(v) == "Empty VData '8' (6 obs x 0 vars over 1 time point).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'data', 'data_bis'\n" \
                      "\ttimepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=None, timepoints=timepoints,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=9)
    assert repr(v) == "Empty VData '9' ([4, 2] obs x 0 vars over 2 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'data', 'data_bis'\n" \
                      "\ttimepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time-points is None
    v = vdata.VData(data=None, obs=obs, var=var, timepoints=None, name=10)
    assert repr(v) == "VData '10' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=var, timepoints=timepoints,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=11)
    assert repr(v) == "VData '11' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data, time_list=["0h", "0h", "0h", "0h", "5h", "5h"])

    #       var is None
    #           time-points is None
    v = vdata.VData(data=None, obs=obs, var=None, timepoints=None, name=12)
    assert repr(v) == "Empty VData '12' ([4, 2] obs x 0 vars over 2 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'data', 'data_bis'\n" \
                      "\ttimepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=None, timepoints=timepoints, name=13)
    assert repr(v) == "Empty VData '13' ([4, 2] obs x 0 vars over 2 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time-points is None
    v = vdata.VData(data=None, obs=obs, var=var, timepoints=None, name=14)
    assert repr(v) == "VData '14' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=var, timepoints=timepoints, name=15)
    assert repr(v) == "VData '15' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #   obs is invalid
    obs = 0

    with pytest.raises(vdata.VTypeError) as exc_info:
        _ = vdata.VData(data=None, obs=obs, var=None, timepoints=None, name=16)

    assert exc_info.value.msg == "'obs' must be a pandas DataFrame or a TemporalDataFrame."

    # data is a pd.DataFrame
    data = pd.DataFrame(expr_data_simple, index=obs_index_data, columns=var_index_data)

    #   obs is None
    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=None, var=None, timepoints=None, name=17)
    assert repr(v) == "VData '17' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    timepoints = pd.DataFrame(timepoints_data_simple)

    v = vdata.VData(data=data, obs=None, var=None, timepoints=timepoints, name=18)
    assert repr(v) == "VData '18' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time-points is None
    v = vdata.VData(data=data, obs=None, var=var, timepoints=None, name=19)
    assert repr(v) == "VData '19' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=None, var=var, timepoints=timepoints, name=20)
    assert repr(v) == "VData '20' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=None, name=21)
    assert repr(v) == "VData '21' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=timepoints, name=22)
    assert repr(v) == "VData '22' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=None, name=23)
    assert repr(v) == "VData '23' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=timepoints, name=24)
    assert repr(v) == "VData '24' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=None, name=25)
    assert repr(v) == "VData '25' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=timepoints, name=26)
    assert repr(v) == "VData '26' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=None, name=27)
    assert repr(v) == "VData '27' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=timepoints, name=28)
    assert repr(v) == "VData '28' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    # data is a dict[str, np.array]
    data = expr_data_complex

    #   obs is None
    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=None, var=None, timepoints=None, name=29)
    assert repr(v) == "VData '29' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    timepoints = pd.DataFrame(timepoints_data)

    v = vdata.VData(data=data, obs=None, var=None, timepoints=timepoints, name=30)
    assert repr(v) == "VData '30' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           timepoints is None
    v = vdata.VData(data=data, obs=None, var=var, timepoints=None, name=31)
    assert repr(v) == "VData '31' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=None, var=var, timepoints=timepoints, name=32)
    assert repr(v) == "VData '32' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=None, time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                    name=33)
    assert repr(v) == "VData '33' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=timepoints,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=34)
    assert repr(v) == "VData '34' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=None, time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                    name=35)
    assert repr(v) == "VData '35' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=timepoints,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=36)
    assert repr(v) == "VData '36' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", \
        repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data, time_list=["0h", "0h", "0h", "0h", "5h", "5h"])

    #       var is None
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=None, name=37)
    assert repr(v) == "VData '37' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, timepoints=timepoints, name=38)
    assert repr(v) == "VData '38' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "timepoints: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time-points is None
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=None, name=39)
    assert repr(v) == "VData '39' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    #           time-points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, timepoints=timepoints, name=40)
    assert repr(v) == "VData '40' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'", repr(v)

    # data is invalid
    data = 0

    with pytest.raises(vdata.VTypeError) as exc_info:
        vdata.VData(data=data, obs=None, var=None, timepoints=None, name=41)

    assert exc_info.value.msg == "Type '<class 'int'>' is not allowed for 'data' parameter, should be a dict," \
        "a pandas DataFrame, a TemporalDataFrame or an AnnData object."


def test_VData_creation_with_uns():
    timepoints = pd.DataFrame({"value": ["0h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "0h", "0h"])
    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    data = pd.DataFrame(expr_data_simple)

    v = vdata.VData(data, timepoints=timepoints, obs=obs, var=var, uns=uns, name=46)
    assert repr(v) == "VData '46' with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'\n\t" \
                      "uns: 'colors', 'date'", repr(v)


def test_VData_creation_full():
    timepoints = pd.DataFrame({"value": ['0h', '5h']})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                  index=obs_index_data, name='obs')
    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    obsm = {'umap': pd.DataFrame({'X1': [4, 5, 6, 7, 8, 9], 'X2': [1, 2, 3, 9, 8, 7]}, index=obs_index_data),
            'pca': pd.DataFrame({'X1': [-4, -5, -6, -7, -8, -9], 'X2': [-1, -2, -3, -9, -8, -7]}, index=obs_index_data)}
    obsp = {'pair': np.array([[1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [0, 0, 0, 0, 2, 2],
                              [0, 0, 0, 0, 2, 2]])}
    varm = {'test': pd.DataFrame({'col': [7, 8, 9]}, index=["g1", "g2", "g3"])}
    varp = {'test2': pd.DataFrame({'g1': [0, 0, 1], 'g2': [0, 1, 0], 'g3': [1, 0, 0]}, index=["g1", "g2", "g3"])}

    v = vdata.VData(expr_data_complex, timepoints=timepoints, obs=obs, var=var, uns=uns,
                    obsm=obsm, obsp=obsp, varm=varm, varp=varp,
                    name=47)

    assert repr(v) == "VData '47' with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'\n\t" \
                      "obsm: 'umap', 'pca'\n\t" \
                      "varm: 'test'\n\t" \
                      "obsp: 'pair'\n\t" \
                      "varp: 'test2'\n\t" \
                      "uns: 'colors', 'date'", repr(v)

    v.set_obs_index([f"C_{i}" for i in range(6, 12)])

    assert repr(v.obsp['pair']) == '      C_6  C_7  C_8  C_9  C_10  C_11\n' \
                                   'C_6     1    1    1    1     0     0\n' \
                                   'C_7     1    1    1    1     0     0\n' \
                                   'C_8     1    1    1    1     0     0\n' \
                                   'C_9     1    1    1    1     0     0\n' \
                                   'C_10    0    0    0    0     2     2\n' \
                                   'C_11    0    0    0    0     2     2', repr(v.obsp['pair'])


def test_VData_creation_from_dict():
    v = vdata.read_from_dict(data, name='1')

    assert repr(v) == "VData '1' with n_obs x n_var = [7, 3, 10] x 4 over 3 time points.\n\t" \
                      "layers: 'RNA', 'Protein'\n\t" \
                      "timepoints: 'value'", repr(v)

    timepoints = pd.DataFrame({"value": ['0h', '5h', '10h']})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3", "g4"]}, index=["g1", "g2", "g3", "g4"])
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 20),
                                   'data_bis': np.random.randint(0, 20, 20)},
                                  time_list=["0h" for _ in range(7)] + ["5h" for _ in range(3)] + ["10h" for _ in
                                                                                                   range(10)],
                                  index=[f"C_{i}" for i in range(20)], name='obs')

    v = vdata.read_from_dict(data, obs=obs, var=var, timepoints=timepoints, name='2')
    assert repr(v) == "VData '2' with n_obs x n_var = [7, 3, 10] x 4 over 3 time points.\n\t" \
                      "layers: 'RNA', 'Protein'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "timepoints: 'value'"


def test_VData_creation_repeating_index():
    obs = vdata.TemporalDataFrame(data={'col1': np.arange(9)},
                                  time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                  index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                                  repeating_index=True)

    data = vdata.TemporalDataFrame(data={'G1': [1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                         'G2': [1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                         'G3': [1., 2., 3., 4., 5., 6., 7., 8., 9.]},
                                   time_list=['0h', '0h', '0h', '5h', '5h', '5h', '10h', '10h', '10h'],
                                   index=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'],
                                   repeating_index=True)

    v = vdata.VData(data, obs=obs)
    assert repr(v) == "VData 'No_Name' with n_obs x n_var = [3, 3, 3] x 3 over 3 time points.\n"\
                      "\tlayers: 'data'\n"\
                      "\tobs: 'col1'\n"\
                      "\ttimepoints: 'value'"


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    test_VData_creation()
    test_VData_creation_with_uns()
    test_VData_creation_full()
    test_VData_creation_from_dict()
