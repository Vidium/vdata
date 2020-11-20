# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from .._core.vdata import VData

# ====================================================
# code

# expr_matrix = pd.DataFrame({"a": [0, 10, 0, 15], "b": [10, 0, 9, 2], "c": [20, 15, 16, 16]}, index=[1, 2, 3, 4])
expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
obsm = {'umap': np.zeros((1, 4, 2))}
obsp = {'connect': np.zeros((4, 4))}
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
varm = None
varp = {'correlation': np.zeros((3, 3))}
layers = {'spliced': np.zeros((1, 4, 3))}
uns = {'color': ["#c1c1c1"]}
time_points = pd.DataFrame({"value": [5], "unit": ["hour"]})


a = VData(data=expr_matrix, obs=obs, obsm=obsm, var=var, varm=varm, uns=uns, time_points=time_points, log_level="DEBUG", dtype="float64")
print(a)

a.obsm = {"test": np.zeros((1, 4, 2))}
a.layers = {"spliced": np.array((8, 8))}

print(a)
print(a.layers)

# a._varm['test'] = np.zeros((2, 2, 2))

# a.write("/home/matteo/Desktop/test.p")
