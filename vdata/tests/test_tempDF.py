# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from .._core.dataframe import TemporalDataFrame
from .._core.vdata import VData
from .._IO.logger import generalLogger

generalLogger.level = 'DEBUG'

# ====================================================
# code
data = {'TP': ['0', '0', '0', '0', '0'] + ['1'],
        'data': np.random.randint(0, 20, 6)}

obs = TemporalDataFrame(data, time_col='TP')

print(obs)
vobs = obs.loc[[5]]
print(vobs)
quit()

# TODO : still a bug here :

print(obs)

print(obs[1]['data'])
print(obs[1]['data'].iat[0, 0])
obs[1]['data'].iat[0, 0] = -1
print(obs[1]['data'])
print(obs[1]['data'].iat[0, 0])

print(obs)

quit()


data = {'TP': [(0, 1)] + list(np.concatenate((np.array(['*']), np.repeat(0, 8), np.repeat(1, 5), np.repeat(2,
                                                                                                           2)))) + [3],
        'ID': [f'C0.{i}' for i in range(10)] + [f'C1.{i}' for i in range(5)] + [f'C2.{i}' for i in range(2)] + ['C3.0'],
        'data': np.random.randint(0, 20, 18),
        'data_bis': 0,
        'data_tris': 0}

# TODO : warn in wiki that all TP are converted to strings
obs = TemporalDataFrame(data, time_points=data['TP'], time_col=None, columns=['TP', 'ID', 'data', 'data_bis'])
print(obs)
quit()

print(obs[3])
print(obs[3].at[17, 'data'])
obs[3].at[17, 'data'] = 30
print(obs[3])
print(obs[3].at[17, 'data'])

# print(obs[0:2, obs.data > 5][0])
# print(obs[0:2, obs.data > 5][0].head())

# obs.data_bis = 1
# obs[0] = pd.DataFrame({'TP': obs[0].TP, 'ID': obs[0].ID, 'data': np.random.randint(0, 20, 10), 'data_bis': 1},
#                       columns=['TP', 'ID', 'data', 'data_bis'])

# print(obs[0, obs.data > 5])

# print(obs.loc[:, ['ID', 'data']])
# print(obs.iloc[0])
# print(list(obs.keys()))
# print(obs.eq(0))

# print(obs[0, obs.data > 7])

# TODO : warn in wiki that you should not use tuples
# print(obs[(0, 1), ])
quit()
# ----------------------------------------------------------------
expr_matrix = np.array([np.zeros((4, 3)), np.zeros((2, 3))], dtype=object)

# TODO : not OK : shapes do not match !
v = VData(data=expr_matrix, obs=obs)
print(v)
