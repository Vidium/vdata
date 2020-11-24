# coding: utf-8
# Created on 11/20/20 11:04 AM
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
import pandas as pd

from .._core.vdata import VData

# ====================================================
# code
os.system('rm -rf /home/matteo/Desktop/vdata')


expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2], "cat": pd.Series(["a", "b", "c", 1], dtype="category", index=[1, 2, 3, 4])}, index=[1, 2, 3, 4])
obsm = {'umap': np.zeros((1, 4, 2))}
obsp = {'connect': np.zeros((4, 4))}
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
varm = None
varp = {'correlation': np.zeros((3, 3))}
layers = {'spliced': np.zeros((1, 4, 3))}
uns = {'color': ["#c1c1c1"], 'str': "test string", "int": 2}
time_points = pd.DataFrame({"value": [5], "unit": ["hour"]})


a = VData(data=expr_matrix, obs=obs, obsm=obsm, obsp=obsp, var=var, varm=varm, varp=varp, uns=uns, time_points=time_points, log_level='DEBUG')
print(a)

a.write("/home/matteo/Desktop/vdata.h5")
