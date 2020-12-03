# coding: utf-8
# Created on 11/17/20 5:19 PM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from .._IO.read import read_from_GPU


# ====================================================
# code
GPU_data = {"RNA": {0: np.zeros((4, 5)),
                    5: np.zeros((4, 5))},
            "Protein": {0: np.zeros((4, 5)),
                        5: np.zeros((4, 5))}}
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2]}, index=[1, 2, 3, 4])
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3", "g4", "g5"]}, index=["a", "b", "c", "d", "e"])
time_points = pd.DataFrame({"value": [0, 5], "unit": ["hour", "hour"]})

a = read_from_GPU(GPU_data, dtype=np.float64)
print(a)

a = read_from_GPU(GPU_data, obs=obs, var=var, time_points=time_points)
print(a)

print('--------------------------')
GPU_data_2 = {"RNA": {"0h": np.zeros((10, 10)),
                      "5h": np.zeros((10, 10))},
              "Protein": {"0h": np.zeros((10, 10)),
                          "5h": np.zeros((10, 10))}}

a = read_from_GPU(GPU_data_2, log_level="DEBUG")
print(a)
print(a.time_points)
