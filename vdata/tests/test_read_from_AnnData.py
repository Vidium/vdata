# coding: utf-8
# Created on 11/20/20 10:06 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

import anndata
from vdata._core import VData

# ====================================================
# code
X = np.array([[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]])
layers = {'spliced': X,
          'unspliced': np.array([[0, 2, 4], [2, 0, 3], [0, 1, 3], [3, 0, 3]])}
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
uns = {'color': ["#c1c1c1"]}
adata = anndata.AnnData(X=X, layers=layers, obs=obs, var=var, uns=uns)
print(adata)

time_points = pd.DataFrame({"value": [0], "unit": ["hour"]})
v = VData(data=adata, time_points=time_points, log_level='DEBUG')
print(v)
