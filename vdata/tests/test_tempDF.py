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
# v = VData(data=expr_matrix, log_level="DEBUG")

data = {'TP': [(0, 1)] + list(np.concatenate((np.array(['*']), np.repeat(0, 8), np.repeat(1, 5), np.repeat(2, 2)))),
        'ID': [f'C0.{i}' for i in range(10)] + [f'C1.{i}' for i in range(5)] + [f'C2.{i}' for i in range(2)],
        'data': np.random.randint(0, 20, 17),
        'data_bis': 0,
        'data_tris': 0}

obs = TemporalDataFrame(data, time_col='TP', columns=['TP', 'ID', 'data', 'data_bis'])
print(obs)

print(obs[0, obs.data > 7])

# TODO : debug this : should be able to do obs[(0, 1)] without ','
# TODO : + does not sub-set as it should !
print(obs[(0, 1), ])




