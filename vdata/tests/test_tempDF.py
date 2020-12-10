# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
import numpy as np

from .._core.dataframe import TemporalDataFrame
from .._core.vdata import VData

# ====================================================
# code
expr_matrix = np.array([np.zeros((4, 3)), np.zeros((2, 3))], dtype=object)
v = VData(data=expr_matrix, log_level="DEBUG")

data = {'TP': [0, 0, 1, 1], 'ID': ['C1.1', 'C1.2', 'C2.1', 'C2.2'], 'data': [2, 10, 5, 15]}

obs = TemporalDataFrame(v, data, time_col='TP')
print(obs)

print(obs[:, obs.data > 7])




