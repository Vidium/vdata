# coding: utf-8
# Created on 12/10/20 9:30 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from _core.vdata import VData


# ====================================================
# code
expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
                        [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
                        [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"],
                    "batch": [1, 1, 2, 2],
                    "cat": pd.Series(["a", "b", "c", 1], dtype="category", index=[1, 2, 3, 4])}, index=[1, 2, 3, 4])
obsm = {'umap': np.zeros((3, 4, 2))}
obsp = {'connect': np.zeros((4, 4))}
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
varm = None
varp = {'correlation': np.zeros((3, 3))}
layers = {'spliced': np.zeros((3, 4, 3))}
uns = {'color': ["#c1c1c1"], 'str': "test string", "int": 2}
time_points = pd.DataFrame({"value": [5, 10, 15], "unit": ["hour", "hour", "hour"]})

a = VData(data=expr_matrix, var=var, varm=varm, varp=varp, uns=uns, time_points=time_points, log_level='DEBUG')
print(a[:])
v = a[:2, (1, 4), ('a', 'c')]
print(v)

b = v.copy()
print(b)
