# coding: utf-8
# Created on 11/25/20 3:41 PM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
import numpy as np

import vdata


# ====================================================
# code
def test_VData_simple_subset():
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col='Time_hour')

    print(v['0'])


def test_VData_sub_setting():
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col='Time_hour')

    print('\n==============================================================================\n')

    mask_obs = ['plate1_A01_A01_B01', 'plate1_A01_A01_E01', 'plate1_A01_A01_H01', 'plate1_A03_A03_A11',
                'plate1_A04_A04_H01', 'plate1_A02_A02_B11', 'plate1_A03_A03_B02', 'plate1_B06_B06_D02',
                'plate1_B10_B10_E02', 'plate1_B12_B12_B02']

    mask_var = ['ENSG00000255794.7', 'ENSG00000276644.4', 'ENSG00000283436.1', 'ENSG00000284600.1']

    sub_vdata = v[:, mask_obs, mask_var]

    print(sub_vdata)
    print(sub_vdata.obs)
    print(sub_vdata.layers['data'])

    print(sub_vdata.n_obs)

    assert np.sum(sub_vdata.n_obs) == len(mask_obs)
    assert np.sum(sub_vdata.n_var) == len(mask_var)

    print(sub_vdata.time_points)

    layers_shape = (sub_vdata.layers['data'].shape[0],
                    [sub_vdata.layers['data'][TP].shape[1][0] for TP in sub_vdata.time_points.value.values],
                    sub_vdata.layers['data'][0].shape[2])
    print(layers_shape)
    assert layers_shape == (2, sub_vdata.n_obs, sub_vdata.n_var)
    assert layers_shape == (2, [5, 5], 4)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_simple_subset()
    test_VData_sub_setting()
