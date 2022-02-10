# coding: utf-8
# Created on 09/02/2022 17:16
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

import vdata


# ====================================================
# code
obs_index_data = [f"C_{i}" for i in range(6)]

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

time_points = pd.DataFrame({"value": ['0h', '5h']})
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["g1", "g2", "g3"])
obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                               'data_bis': np.random.randint(0, 20, 6)},
                              time_list=["0h", "0h", "0h", "0h", "5h", "5h"],
                              index=obs_index_data, name='obs')
uns = {"colors": ['blue', 'red', 'yellow'],
       "date": '25/01/2021'}

obsm = {'umap': pd.DataFrame({'X1': [4, 5, 6, 7, 8, 9], 'X2': [1, 2, 3, 9, 8, 7]}, index=obs_index_data),
        'pca': pd.DataFrame({'X1': [-4, -5, -6, -7, -8, -9], 'X2': [-1, -2, -3, -9, -8, -7]}, index=obs_index_data)}
obsp = {'pair': np.array([[1, 1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 2, 2],
                          [0, 0, 0, 0, 2, 2]])}
varm = {'test': pd.DataFrame({'col': [7, 8, 9]}, index=["g1", "g2", "g3"])}
varp = {'test2': pd.DataFrame({'g1': [0, 0, 1], 'g2': [0, 1, 0], 'g3': [1, 0, 0]}, index=["g1", "g2", "g3"])}

v = vdata.VData(expr_data_complex, time_points=time_points, obs=obs, var=var, uns=uns,
                obsm=obsm, obsp=obsp, varm=varm, varp=varp,
                name=47)

new_umap_coords = vdata.TemporalDataFrame({'X1': [1, 2, 3, 4, 5, 6],
                                           'X2': [10, 20, 30, 40, 50, 60],
                                           'X3': [100, 200, 300, 400, 500, 600]},
                                          time_list=['0h', '0h', '0h', '0h', '5h', '5h'],
                                          index=obs_index_data)

v.obsm['umap'] = new_umap_coords
