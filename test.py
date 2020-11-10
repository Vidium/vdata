# coding: utf-8
# Created on 11/4/20 10:34 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from .vdata import VData

# ====================================================
# code

expr_matrix = np.array([[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]])
cells = pd.DataFrame({"name": [1, 2, 3, 4]})
genes = pd.DataFrame({"name": ["a", "b", "c"]})

a = VData(X=expr_matrix, obs=cells, var=genes, log_level="INFO", dtype=np.float64)
print(a)

expr_matrix = pd.DataFrame({"a": [0, 10, 0, 15], "b": [10, 0, 9, 2], "c": [20, 15, 16, 16]}, index=[1, 2, 3, 4])
a = VData(X=expr_matrix, log_level="INFO", dtype=np.float64)
print(a.obs)
