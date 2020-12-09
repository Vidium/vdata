# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
from .._core.dataframe import TemporalDataFrame

# ====================================================
# code
data = {'ID': ['C1.1', 'C1.2', 'C2.1', 'C2.2'], 'data': ['a', 'b', 'c', 'd']}
TPID = [0, 0, 1, 1]

obs = TemporalDataFrame(data, TPID)
print(obs)
print(obs[0])
print(obs[:, [True, False, False, True]])
