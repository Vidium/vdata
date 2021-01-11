# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from vdata import setLoggingLevel, TemporalDataFrame, VData
from vdata._IO.errors import VValueError, VTypeError

setLoggingLevel('INFO')


# ====================================================
# code
expr_data_simple = np.array([[10, 11, 12],
                             [20, 21, 22],
                             [30, 31, 32],
                             [40, 41, 42],
                             [50, 51, 52],
                             [60, 61, 62]])

expr_data_medium = {
    "spliced": pd.DataFrame(np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32],
                                      [40, 41, 42], [50, 51, 52], [60, 61, 62]])),
    "unspliced": pd.DataFrame(np.array([[1., 1.1, 1.2], [2., 2.1, 2.2], [3., 3.1, 3.2],
                                        [4., 4.1, 4.2], [5., 5.1, 5.2], [6., 6.1, 6.2]]))
}

expr_data_complex = {
    "spliced": np.array([
        np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]),
        np.array([[50, 51, 52], [60, 61, 62]])
    ], dtype=object),
    "unspliced": np.array([
        np.array([[1., 1.1, 1.2], [2., 2.1, 2.2], [3., 3.1, 3.2], [4., 4.1, 4.2]]),
        np.array([[5., 5.1, 5.2], [6., 6.1, 6.2]])
    ], dtype=object)
}


def test_object_creation():
    time_points_data = {"value": [0, 5], "unit": "hour"}
    time_points_data_incorrect_format = {"no_column_value": [0, 5], "unit": "hour"}
    time_points_data_simple = {"value": [0], "unit": "hour"}

    var_data = {"gene_name": ["g1", "g2", "g3"]}
    var_index_data = ["a", "b", "c"]

    obs_data = {'data': np.random.randint(0, 20, 6),
                'data_bis': np.random.randint(0, 20, 6)}
    obs_index_data = [f"C_{i}" for i in range(6)]

    # data is None
    #   obs is None
    #       var is None
    #           time_points is None
    v = VData(data=None, obs=None, var=None, time_points=None)
    assert repr(v) == "Empty Vdata object (0 obs x 0 vars over 0 time point)."

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = VData(data=None, obs=None, var=None, time_points=time_points)
    assert repr(v) == "Empty Vdata object (0 obs x 0 vars over 2 time points).\n\ttime_points: 'value', 'unit'"

    time_points = pd.DataFrame(time_points_data_incorrect_format)

    try:
        VData(data=None, obs=None, var=None, time_points=time_points)

    except VValueError as e:
        assert e.msg == "'time points' must have at least a column 'value' to store time points value."

    #           time_points is invalid
    time_points = 0

    try:
        VData(data=None, obs=None, var=None, time_points=time_points)

    except VTypeError as e:
        assert e.msg == "'time points' must be a pandas DataFrame."

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = VData(data=None, obs=None, var=var, time_points=None)
    assert repr(v) == "Empty Vdata object (0 obs x 3 vars over 0 time point).\n\tvar: 'gene_name'"

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = VData(data=None, obs=None, var=var, time_points=time_points)
    assert repr(v) == "Empty Vdata object (0 obs x 3 vars over 2 time points).\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value', 'unit'"

    #       var is invalid
    var = 0

    try:
        v = VData(data=None, obs=None, var=var, time_points=None)

    except VTypeError as e:
        assert e.msg == "var must be a pandas DataFrame."

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=None, obs=obs, var=None, time_points=None)
    assert repr(v) == "Empty Vdata object (6 obs x 0 vars over 0 time point).\n\tobs: 'data', 'data_bis'"

    #           time_points is a pd.DataFrame
    v = VData(data=None, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Empty Vdata object (6 obs x 0 vars over 2 time points).\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = VData(data=None, obs=obs, var=var, time_points=None)
    assert repr(v) == "Empty Vdata object (6 obs x 3 vars over 0 time point).\n\tobs: 'data', 'data_bis'\n\tvar: " \
                      "'gene_name'"

    #           time_points is a pd.DataFrame
    v = VData(data=None, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Empty Vdata object (6 obs x 3 vars over 2 time points).\n\tobs: 'data', 'data_bis'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a TemporalDataFrame
    obs = TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=None, obs=obs, var=None, time_points=None)
    assert repr(v) == "Empty Vdata object (6 obs x 0 vars over 0 time point).\n\tobs: 'data', 'data_bis'"

    #           time_points is a pd.DataFrame
    v = VData(data=None, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Empty Vdata object (6 obs x 0 vars over 2 time points).\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    var = pd.DataFrame(var_data, index=var_index_data)

    #           time_points is None
    v = VData(data=None, obs=obs, var=var, time_points=None)
    assert repr(v) == "Empty Vdata object (6 obs x 3 vars over 0 time point).\n\tobs: 'data', 'data_bis'\n\tvar: " \
                      "'gene_name'"

    #           time_points is a pd.DataFrame
    v = VData(data=None, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Empty Vdata object (6 obs x 3 vars over 2 time points).\n\tobs: 'data', 'data_bis'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is invalid
    obs = 0

    try:
        v = VData(data=None, obs=obs, var=None, time_points=None)

    except VTypeError as e:
        assert e.msg == "obs must be a pandas DataFrame or a TemporalDataFrame."

    # data is a pd.DataFrame
    data = pd.DataFrame(expr_data_simple, index=obs_index_data, columns=var_index_data)

    #   obs is None
    #       var is None
    #           time_points is None
    v = VData(data=data, obs=None, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\ttime_points: " \
                      "'value'"

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data_simple)

    v = VData(data=data, obs=None, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\ttime_points: " \
                      "'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=None, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a TemporalDataFrame
    obs = TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    # data is a np.array
    data = expr_data_simple

    #   obs is None
    #       var is None
    #           time_points is None
    v = VData(data=data, obs=None, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\ttime_points: " \
                      "'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\ttime_points: " \
                      "'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=None, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tvar: " \
                      "'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a TemporalDataFrame
    obs = TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'data'\n\tobs: 'data', " \
                      "'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    # data is a dict[str, pd.DataFrame]
    data = expr_data_medium

    #   obs is None
    #       var is None
    #           time_points is None
    v = VData(data=data, obs=None, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=None, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a TemporalDataFrame
    obs = TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = 6 x 3 over 1 time point\n\tlayers: 'spliced', 'unspliced'" \
                      "\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    # data is a dict[str, np.array]
    data = expr_data_complex

    #   obs is None
    #       var is None
    #           time_points is None
    v = VData(data=data, obs=None, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    time_points = pd.DataFrame(time_points_data)

    v = VData(data=data, obs=None, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=None, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=None, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a pd.DataFrame
    obs = pd.DataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    #   obs is a TemporalDataFrame
    obs = TemporalDataFrame(obs_data, index=obs_index_data)

    #       var is None
    #           time_points is None
    v = VData(data=data, obs=obs, var=None, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=None, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\ttime_points: 'value', 'unit'"

    #       var is a pd.DataFrame
    #           time_points is None
    v = VData(data=data, obs=obs, var=var, time_points=None)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value'"

    #           time_points is a pd.DataFrame
    v = VData(data=data, obs=obs, var=var, time_points=time_points)
    assert repr(v) == "Vdata object with n_obs x n_var = [4, 2] x 3 over 2 time points\n\tlayers: 'spliced', " \
                      "'unspliced'\n\tobs: 'data', 'data_bis'\n\tvar: 'gene_name'\n\ttime_points: 'value', 'unit'"

    # data is invalid
    data = 0

    try:
        VData(data=data, obs=None, var=None, time_points=None)

    except VTypeError as e:
        assert e.msg == "Type '<class 'int'>' is not allowed for 'data' parameter, should be a dict,a pandas " \
                        "DataFrame, a numpy array or an AnnData object."


def test_object_creation_on_dtype():
    time_points = pd.DataFrame({"value": [0], "unit": "hour"})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                             'data_bis': np.random.randint(0, 20, 6)})

    # dtype is invalid
    try:
        VData(expr_data_complex, time_points=time_points, obs=obs, var=var, dtype="NOT A DATA TYPE")

    except VTypeError as e:
        assert e.msg == "Incorrect data-type 'NOT A DATA TYPE', should be in [<class 'int'>, 'int', 'int8', 'int16', " \
                        "'int32', 'int64', <class 'float'>, 'float', 'float16', 'float32', 'float64', 'float128', " \
                        "<class 'numpy.int64'>, <class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, " \
                        "<class 'numpy.float64'>, <class 'numpy.float16'>, <class 'numpy.float32'>, " \
                        "<class 'numpy.float128'>, <class 'object'>]"

    # data is a pd.DataFrame
    data = pd.DataFrame(expr_data_simple)

    v = VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128)
    assert v.layers['data'].dtype == np.float128

    # data is a np.array
    data = expr_data_simple

    v = VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128)
    assert v.layers['data'].dtype == np.float128

    # data is a dict[str, pd.DataFrame]
    data = expr_data_medium

    v = VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128)
    assert v.layers['spliced'].dtype == np.float128

    # data is a dict[str, np.array]
    data = expr_data_complex
    time_points = pd.DataFrame({"value": [0, 5], "unit": "hour"})

    v = VData(data, time_points=time_points, obs=obs, var=var, dtype=np.float128)
    assert v.layers['spliced'][0].dtype == np.float128


test_object_creation()
test_object_creation_on_dtype()


# quit()
#
#
#
#
#
# expr_matrix = {
#     "spliced": np.array([
#         np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]),
#         np.array([[50, 51, 52], [60, 61, 62]])
#     ], dtype=object),
#     "unspliced": np.array([
#         np.array([[1., 1.1, 1.2], [2., 2.1, 2.2], [3., 3.1, 3.2], [4., 4.1, 4.2]]),
#         np.array([[5., 5.1, 5.2], [6., 6.1, 6.2]])
#     ], dtype=object)
# }
#
# time_points = pd.DataFrame({"value": [0, 5], "unit": "hour"})
# var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
#
# data_obs = {'data': np.random.randint(0, 20, 6),
#             'data_bis': np.random.randint(0, 20, 6)}
#
# # obs is None
# v = VData(data=expr_matrix, var=var, time_points=time_points)
# print(v)
#
# # obs is a pandas DataFrame
# obs = pd.DataFrame(data_obs, index=[f'C_{i}' for i in range(6)])
#
# v = VData(data=expr_matrix, obs=obs, var=var, time_points=time_points)
# print(v)
#
# # obs is a TemporalDataFrame
# obs = TemporalDataFrame(data_obs, index=[f'C_{i}' for i in range(6)], time_points=['0', '0', '0', '0', '5', '5'])
#
# v = VData(data=expr_matrix, obs=obs, var=var, time_points=time_points)
# print(v)
#
#
#
#
# quit()
#
# time_points = pd.DataFrame({"value": [0, 5], "unit": "hour"})
#
# data_obs = {'data': np.random.randint(0, 20, 6),
#             'data_bis': np.random.randint(0, 20, 6)}
#
# # TODO : check length of obs is OK, check time points are OK
# obs = TemporalDataFrame(data_obs, index=[f'C_{i}' for i in range(6)], time_points=[0, 0, 0, 0, 5, 5])
#
# var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
#
# v = VData(data=expr_matrix, obs=obs, var=var, time_points=time_points)
# print(v)
#
# print(v[5])
# print(v[[0, 5]].layers)
#
# v.write("/home/matteo/Desktop/v.h5")

# expr_matrix = pd.DataFrame({"a": [0, 10, 0, 15], "b": [10, 0, 9, 2], "c": [20, 15, 16, 16]}, index=[1, 2, 3, 4])
# expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
# obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
# obsm = {'umap': np.zeros((1, 4, 2))}
# obsp = {'connect': np.zeros((4, 4))}
# var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
# varm = None
# varp = {'correlation': np.zeros((3, 3))}
# layers = {'spliced': np.zeros((1, 4, 3))}
# uns = {'color': ["#c1c1c1"]}
# time_points = pd.DataFrame({"value": [5], "unit": ["hour"]})
#
#
# VData(data=expr_matrix, obs=obs, obsm=obsm, var=var, varm=varm, uns=uns, time_points=time_points, dtype="float64")
