# coding: utf-8
# Created on 12/9/20 10:21 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from vdata import TemporalDataFrame, setLoggingLevel

setLoggingLevel('DEBUG')


# ====================================================
# code
TDF = TemporalDataFrame(pd.DataFrame({'test': [5]}), time_list=['*'])
print(TDF)
print(TDF['0'])


TDF = TemporalDataFrame(pd.DataFrame({'test': [5]}), time_list=['*'], time_points=[0, 1])
print(TDF)


data = {'data': np.random.randint(0, 20, 7),
        'data_bis': np.random.randint(0, 20, 7),
        'TP': ['*', '1', ('0', '1'), '0', '0', '1', '0']}

TDF = TemporalDataFrame(data, time_col='TP', index=[f"C_{i}" for i in range(7)])
print(TDF)
