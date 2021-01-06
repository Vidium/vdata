# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from vdata import setLoggingLevel, TemporalDataFrame, VData

setLoggingLevel('INFO')

# ====================================================
# code

expr_matrix = {
    "spliced": np.array([
        np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]),
        np.array([[50, 51, 52], [60, 61, 62]])
    ], dtype=object),
    "unspliced": np.array([
        np.array([[1., 1.1, 1.2], [2., 2.1, 2.2], [3., 3.1, 3.2], [4., 4.1, 4.2]]),
        np.array([[5., 5.1, 5.2], [6., 6.1, 6.2]])
    ], dtype=object)
}

time_points = pd.DataFrame({"value": [0, 5], "unit": "hour"})

data_obs = {'data': np.random.randint(0, 20, 6),
            'data_bis': np.random.randint(0, 20, 6)}

# TODO : check length of obs is OK, check time points are OK
obs = TemporalDataFrame(data_obs, index=[f'C_{i}' for i in range(6)], time_points=[0, 0, 0, 0, 5, 5])

var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])

v = VData(data=expr_matrix, obs=obs, var=var, time_points=time_points)
print(v)

print(v[5])
print(v[[0, 5]].layers)

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
