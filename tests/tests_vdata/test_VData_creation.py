# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

import vdata
from . import obs_index_data, expr_data_simple, expr_data_medium, expr_data_complex


# ====================================================
# code
def test_VData_creation():
    time_points_data = {"value": ["0h", "5h"]}
    time_points_data_incorrect_format = {"no_column_value": ["0h", "5h"]}
    time_points_data_simple = {"value": ["0h"]}

    var_data = {"gene_name": ["gene 1", "gene 2", "gene 3"]}
    var_index_data = ["g1", "g2", "g3"]

    obs_data = {'data': np.random.randint(0, 20, 6),
                'data_bis': np.random.randint(0, 20, 6)}

    # data is None
    #   obs is None
    #       var is None
    #           time_points is None
    v = vdata.VData(data=None, obs=None, var=None, time_points=None, name=1)
    assert repr(v) == "Empty Vdata object (0 obs x 0 vars over 0 time points).", repr(v)

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = vdata.VData(data=None, obs=None, var=None, time_points=time_points, name=2)
    assert repr(v) == "Empty Vdata object ([0, 0] obs x 0 vars over 2 time points).\n\ttime_points: 'value'", \
        repr(v)

    time_points = pd.DataFrame(time_points_data_incorrect_format)

    try:
        vdata.VData(data=None, obs=None, var=None, time_points=time_points, name=3)

    except vdata.VValueError as e:
        assert e.msg == "'time points' must have at least a column 'value' to store time points value.", repr(v)

    #           time_points is invalid
    time_points = 0

    try:
        vdata.VData(data=None, obs=None, var=None, time_points=time_points, name=4)

    except vdata.VTypeError as e:
        assert e.msg == "'time points' must be a pandas DataFrame."

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = vdata.VData(data=None, obs=None, var=var, time_points=None, name=5)
    assert repr(v) == "Empty Vdata object (0 obs x 3 vars over 0 time points).\n\tvar: 'gene_name'", repr(v)

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = vdata.VData(data=None, obs=None, var=var, time_points=time_points, name=6)
    assert repr(v) == "Empty Vdata object ([0, 0] obs x 3 vars over 2 time points).\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is invalid
    var = 0

    try:
        _ = vdata.VData(data=None, obs=None, var=var, time_points=None, name=7)

    except vdata.VTypeError as e:
        assert e.msg == "var must be a pandas DataFrame."

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = vdata.VData(data=None, obs=obs, var=None, time_points=None, name=8)
    assert repr(v) == "Empty Vdata object (6 obs x 0 vars over 1 time point).\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=None, time_points=time_points,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=9)
    assert repr(v) == "Empty Vdata object ([4, 2] obs x 0 vars over 2 time points).\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = vdata.VData(data=None, obs=obs, var=var, time_points=None, name=10)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=var, time_points=time_points,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=11)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data, time_list=["0h", "0h", "0h", "0h", "5h", "5h"])

    #       var is None
    #           time_points is None
    v = vdata.VData(data=None, obs=obs, var=None, time_points=None, name=12)
    assert repr(v) == "Empty Vdata object ([4, 2] obs x 0 vars over 2 time points).\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=None, time_points=time_points, name=13)
    assert repr(v) == "Empty Vdata object ([4, 2] obs x 0 vars over 2 time points).\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = vdata.VData(data=None, obs=obs, var=var, time_points=None, name=14)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=None, obs=obs, var=var, time_points=time_points, name=15)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #   obs is invalid
    obs = 0

    try:
        _ = vdata.VData(data=None, obs=obs, var=None, time_points=None, name=16)

    except vdata.VTypeError as e:
        assert e.msg == "'obs' must be a pandas DataFrame or a TemporalDataFrame."

    # data is a pd.DataFrame
    data = pd.DataFrame(expr_data_simple, index=obs_index_data, columns=var_index_data)

    #   obs is None
    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=None, var=None, time_points=None, name=17)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data_simple)

    v = vdata.VData(data=data, obs=None, var=None, time_points=time_points, name=18)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=None, var=var, time_points=None, name=19)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=None, var=var, time_points=time_points, name=20)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=None, time_points=None, name=21)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, time_points=time_points, name=22)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=var, time_points=None, name=23)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, time_points=time_points, name=24)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=None, time_points=None, name=25)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, time_points=time_points, name=26)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=var, time_points=None, name=27)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, time_points=time_points, name=28)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    # data is a dict[str, np.array]
    data = expr_data_complex

    #   obs is None
    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=None, var=None, time_points=None, name=29)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = vdata.VData(data=data, obs=None, var=None, time_points=time_points, name=30)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=None, var=var, time_points=None, name=31)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=None, var=var, time_points=time_points, name=32)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=None, time_points=None, time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                    name=33)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, time_points=time_points,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=34)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=var, time_points=None, time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                    name=35)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, time_points=time_points,
                    time_list=["0h", "0h", "0h", "0h", "5h", "5h"], name=36)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", \
        repr(v)

    #   obs is a TemporalDataFrame
    obs = vdata.TemporalDataFrame(obs_data, index=obs_index_data, time_list=["0h", "0h", "0h", "0h", "5h", "5h"])

    #       var is None
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=None, time_points=None, name=37)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=None, time_points=time_points, name=38)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "time_points: 'value'", repr(v)

    #       var is a pd.DataFrame
    #           time_points is None
    v = vdata.VData(data=data, obs=obs, var=var, time_points=None, name=39)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    #           time_points is a pd.DataFrame
    v = vdata.VData(data=data, obs=obs, var=var, time_points=time_points, name=40)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points.\n\t" \
                      "layers: 'spliced', 'unspliced'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'", repr(v)

    # data is invalid
    data = 0

    try:
        vdata.VData(data=data, obs=None, var=None, time_points=None, name=41)

    except vdata.VTypeError as e:
        assert e.msg == "Type '<class 'int'>' is not allowed for 'data' parameter, should be a dict," \
                        "a pandas DataFrame, a TemporalDataFrame or an AnnData object."


def test_VData_creation_on_dtype():
    time_points = pd.DataFrame({"value": ["0h"]})
    var = pd.DataFrame({"gene_name": ["gene 1", "gene 2", "gene 3"]}, index=['g1', 'g2', 'g3'])
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "0h", "0h"])

    # dtype is invalid
    try:
        vdata.VData(expr_data_complex, time_points=time_points, obs=obs, var=var, dtype="NOT A DATA TYPE", name=42)

    except vdata.VTypeError as e:
        assert e.msg == "Incorrect data-type 'NOT A DATA TYPE', should be in [<class 'int'>, 'int', 'int8', 'int16', " \
                        "'int32', 'int64', <class 'float'>, 'float', 'float16', 'float32', 'float64', 'float128', " \
                        "<class 'numpy.int64'>, <class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, " \
                        "<class 'numpy.float64'>, <class 'numpy.float16'>, <class 'numpy.float32'>, " \
                        "<class 'numpy.float128'>, <class 'object'>]"

    # data is a pd.DataFrame
    data = pd.DataFrame(expr_data_simple, columns=['g1', 'g2', 'g3'])

    v = vdata.VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128, name=43)
    assert v.layers['data'].dtypes.equals(pd.Series([np.float128, np.float128, np.float128],
                                                    index=['g1', 'g2', 'g3'])), v.layers['data'].dtypes

    # data is a dict[str, pd.DataFrame]
    data = expr_data_medium

    v = vdata.VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128, name=44)
    assert v.layers['spliced'].dtypes.equals(pd.Series([np.float128, np.float128, np.float128],
                                                       index=['g1', 'g2', 'g3'])), \
        v.layers['spliced'].dtypes

    # data is a dict[str, np.array]
    data = expr_data_complex
    time_points = pd.DataFrame({"value": ["0h", "5h"]})
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                  index=obs_index_data)

    v = vdata.VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128, name=45)
    assert v.layers['spliced'].dtypes.equals(pd.Series([np.float128, np.float128, np.float128],
                                                       index=['g1', 'g2', 'g3'])), v.layers['spliced'].dtypes


def test_VData_creation_with_uns():
    time_points = pd.DataFrame({"value": ["0h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "0h", "0h"])
    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    data = pd.DataFrame(expr_data_simple)

    v = vdata.VData(data, time_points=time_points, obs=obs, var=var, uns=uns, name=46)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'data', 'data_bis'\n\t" \
                      "var: 'gene_name'\n\t" \
                      "time_points: 'value'\n\t" \
                      "uns: 'colors', 'date'", repr(v)


def test_VData_creation_full():
    time_points = pd.DataFrame({"value": ['0h', '5h']})
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

    v = vdata.VData(expr_data_complex, time_points=time_points, obs=obs, var=var, uns=uns,
                    obsm=obsm, obsp=obsp, varm=varm, varp=varp,
                    name=1)

    print(v.obsp['pair'])

    v.set_obs_index([f"C_{i}" for i in range(6, 12)])

    print(v.obsp['pair'])


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    # test_VData_creation()
    # test_VData_creation_on_dtype()
    # test_VData_creation_with_uns()
    test_VData_creation_full()
