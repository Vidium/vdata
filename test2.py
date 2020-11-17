# coding: utf-8
# Created on 11/17/20 5:19 PM
# Author : matteo

# ====================================================
# imports
import numpy as np

from .IO.read import read_from_GPU


# ====================================================
# code
GPU_data = {"RNA": {0: np.zeros((10, 10)),
                    5: np.zeros((10, 10))},
            "Protein": {0: np.zeros((10, 10)),
                        5: np.zeros((10, 10))}}

a = read_from_GPU(GPU_data, log_level="DEBUG")
print(a)
print(a.time_points)

print('--------------------------')
GPU_data_2 = {"RNA": {"0h": np.zeros((10, 10)),
                      "5h": np.zeros((10, 10))},
              "Protein": {"0h": np.zeros((10, 10)),
                          "5h": np.zeros((10, 10))}}

a = read_from_GPU(GPU_data_2, log_level="DEBUG")
print(a)
print(a.time_points)
