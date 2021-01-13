# coding: utf-8
# Created on 11/25/20 3:41 PM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
import numpy as np
import pandas as pd

from vdata import VData, setLoggingLevel

setLoggingLevel('DEBUG')


# ====================================================
# code
def test_sub_setting():
    # data = {'data': np.array([
    #     np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9],
    #               [10, 11, 12]]),
    #     np.array([[13, 14, 15],
    #               [16, 17, 18],
    #               [19, 20, 21],
    #               [22, 23, 24]]),
    #     np.array([[25, 26, 27],
    #               [28, 29, 30]])
    # ], dtype=object)}
    # obs = pd.DataFrame({"time": [240.0, 0.0, 24.0, 24.0, 0.0, 240.0, 24.0, 0.0, 24.0, 0.0]},
    #                    index=[f'C_{i}' for i in range(10)])
    # var = pd.DataFrame({"val": np.random.randint(1, 10, 3)}, index=[f"G_{i}" for i in range(3)])
    #
    # vdata = VData(data, obs=obs, var=var, time_col='time')

    source_vdata_path = "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    vdata = VData(adata, time_col='Time_hour')

    print('\n==============================================================================\n')

    mask_obs = ['plate1_A01_A01_B01', 'plate1_A01_A01_E01', 'plate1_A01_A01_H01', 'plate1_A03_A03_A11',
                'plate1_A04_A04_H01', 'plate1_A02_A02_B11', 'plate1_A03_A03_B02', 'plate1_B06_B06_D02',
                'plate1_B10_B10_E02', 'plate1_B12_B12_B02']

    mask_var = ['ENSG00000255794.7', 'ENSG00000276644.4', 'ENSG00000283436.1', 'ENSG00000284600.1']

    sub_vdata = vdata[:, mask_obs, mask_var]

    print(sub_vdata)
    print(sub_vdata.obs)
    print(sub_vdata.layers['data'])

    assert np.sum(sub_vdata.n_obs) == len(mask_obs)
    assert np.sum(sub_vdata.n_var) == len(mask_var)

    layers_shape = (sub_vdata.layers['data'].shape[0],
            [sub_vdata.layers['data'][TP].shape[0] for TP in range(sub_vdata.layers['data'].shape[0])],
            sub_vdata.layers['data'][0].shape[1])
    print(layers_shape)
    assert layers_shape == (2, sub_vdata.n_obs, sub_vdata.n_var)



test_sub_setting()




# os.system('rm -rf /home/matteo/Desktop/vdata')
#
#
# expr_matrix = np.array([[[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
#                         [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]],
#                         [[0, 10, 20], [10, 0, 15], [0, 9, 16], [15, 2, 16]]])
# obs = pd.DataFrame({"cell_name": ["c1", "c2", "c3", "c4"],
#                     "batch": [1, 1, 2, 2],
#                     "cat": pd.Series(["a", "b", "c", 1], dtype="category", index=[1, 2, 3, 4])}, index=[1, 2, 3, 4])
# obsm = {'umap': np.zeros((3, 4, 2))}
# obsp = {'connect': np.zeros((4, 4))}
# var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]}, index=["a", "b", "c"])
# varm = None
# varp = {'correlation': np.zeros((3, 3))}
# layers = {'spliced': np.zeros((3, 4, 3))}
# uns = {'color': ["#c1c1c1"], 'str': "test string", "int": 2}
# time_points = pd.DataFrame({"value": [5, 10, 15], "unit": ["hour", "hour", "hour"]})
#
# a = VData(data=expr_matrix, var=var, varm=varm, varp=varp, uns=uns, time_points=time_points, log_level='DEBUG')
# print(a[:])
# v = a[:, (1, 4), ('a', 'c')]
# print(v)
#
# print('----------------------------------------------------')
# a = VData(data=expr_matrix,
#           obs=obs, obsm=obsm, obsp=obsp,
#           var=var, varm=varm, varp=varp,
#           uns=uns, time_points=time_points, log_level='DEBUG')
# print(a)
#
# v = a[1, :, ('a', 'c')]
# print(v)
#
# new_var = pd.DataFrame({'gene_name': ['ng1', 'ng3']})
# print(v.var)
# print(new_var)
# v.var = new_var
#
# print(v.var)
# print(a.var)
#
# print(v.layers['data'])
# v.layers['data'] = np.array([[[-1, 200],
#                               [100, 150],
#                               [-1, 160],
#                               [150, 160]]])
# print(v.layers['data'])
# print(a.layers['data'])
#
# print('----------------------------------------------------')
# v = a[:, (1, 4), ('a', 'c')]
# print(v)
# print(v.obsm['umap'])
# v.obsm['umap'] = np.array([[[1, 2], [-1, -2]], [[3, 4], [-3, -4]], [[5, 6], [-5, -6]]])
# print(v.obsm['umap'])
# print(a.obsm['umap'])
#
# print('----------------------------------------------------')
# print(v.obsp['connect'])
# v.obsp['connect'] = np.array([[1, 2], [3, 4]])
# print(v.obsp['connect'])
# print(a.obsp['connect'])
#
# print('----------------------------------------------------')
# print(v)
# vv = v[:2, 1, 'a']
# print(vv)
#
# print(a[:2, 1:3, ('a', 'b')])
