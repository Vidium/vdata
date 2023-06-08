# coding: utf-8
# Created on 29/01/2021 09:07
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

import vdata

# ====================================================
# code
obs_index_data = [f"C_{i}" for i in range(6)]

expr_data_simple = np.array([[10, 11, 12],
                             [20, 21, 22],
                             [30, 31, 32],
                             [40, 41, 42],
                             [50, 51, 52],
                             [60, 61, 62]])

expr_data_medium = {
    "spliced": pd.DataFrame(np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32],
                                      [40, 41, 42], [50, 51, 52], [60, 61, 62]]),
                            columns=['g1', 'g2', 'g3']),
    "unspliced": pd.DataFrame(np.array([[1., 1.1, 1.2], [2., 2.1, 2.2], [3., 3.1, 3.2],
                                        [4., 4.1, 4.2], [5., 5.1, 5.2], [6., 6.1, 6.2]]),
                              columns=['g1', 'g2', 'g3'])
}

expr_data_complex = {
    "spliced": vdata.TemporalDataFrame({"g1": [10, 20, 30, 40, 50, 60],
                                        "g2": [11, 21, 31, 41, 51, 61],
                                        "g3": [12, 22, 32, 42, 52, 62]},
                                       time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                       index=obs_index_data),

    "unspliced": vdata.TemporalDataFrame({"g1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                          "g2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                                          "g3": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2]},
                                         time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                                         index=obs_index_data)
}
