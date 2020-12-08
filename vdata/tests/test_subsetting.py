# coding: utf-8
# Created on 11/25/20 3:41 PM
# Author : matteo

# ====================================================
# imports
import os
import numpy as np
import pandas as pd

from .._core.vdata import VData

# ====================================================
# code
os.system('rm -rf /home/matteo/Desktop/vdata')


expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
                        [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
                        [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"], "batch": [1, 1, 2, 2], "cat": pd.Series(["a", "b", "c", 1], dtype="category", index=[1, 2, 3, 4])}, index=[1, 2, 3, 4])
obsm = {'umap': np.zeros((3, 4, 2))}
obsp = {'connect': np.zeros((4, 4))}
var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
varm = None
varp = {'correlation': np.zeros((3, 3))}
layers = {'spliced': np.zeros((3, 4, 3))}
uns = {'color': ["#c1c1c1"], 'str': "test string", "int": 2}
time_points = pd.DataFrame({"value": [5, 10, 15], "unit": ["hour", "hour", "hour"]})


a = VData(data=expr_matrix, obs=obs, obsm=obsm, obsp=obsp, var=var, varm=varm, varp=varp, uns=uns, time_points=time_points, log_level='DEBUG')
print(a)

v = a[1, (1, 3)]
print(v)

new_var = pd.DataFrame({'gene_name': ['ng1', 'ng3']})
print(v.var)
print(new_var)
v.var = new_var

print(v.var)
print(a.var)

print(v.layers['data'])
v.layers['data'] = np.array([[[100, 200]], [[300, 400]], [[500, 600]]])
print(v.layers['data'])
print(a.layers['data'])

print('----------------------------------------------------')
v = a[(1, 4), ('a', 'c')]
print(v)
print(v.obsm['umap'])
v.obsm['umap'] = np.array([[[1, 2], [-1, -2]], [[3, 4], [-3, -4]], [[5, 6], [-5, -6]]])
print(v.obsm['umap'])
print(a.obsm['umap'])

print('----------------------------------------------------')
print(v.obsp['connect'])
v.obsp['connect'] = np.array([[1, 2], [3, 4]])
print(v.obsp['connect'])
print(a.obsp['connect'])

print('----------------------------------------------------')
vv = v[1, 'a', :2]
print(vv)

print(a[1:3, ('a', 'b'), :2])
