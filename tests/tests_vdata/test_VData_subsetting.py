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
def test_VData_sub_setting():
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col='Time_hour')

    mask_obs = ['plate1_A01_A01_B01', 'plate1_A01_A01_E01', 'plate1_A01_A01_H01', 'plate1_A03_A03_A11',
                'plate1_A04_A04_H01', 'plate1_A02_A02_B11', 'plate1_A03_A03_B02', 'plate1_B06_B06_D02',
                'plate1_B10_B10_E02', 'plate1_B12_B12_B02']

    mask_var = ['ENSG00000255794.7', 'ENSG00000276644.4', 'ENSG00000187323.11', 'ENSG00000178568.14']

    sub_vdata = v[:, mask_obs, mask_var]

    assert repr(sub_vdata) == "View of a Vdata object with n_obs x n_var = [5, 5] x 4 over 2 time points\n\t" \
                              "layers: 'data'\n\t" \
                              "obs: 'Cell_Type', 'Day', 'Time_hour'\n\t" \
                              "var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n\t" \
                              "time_points: 'value'", repr(sub_vdata)

    assert sub_vdata.n_obs_total == len(mask_obs), sub_vdata.n_obs_total
    assert sub_vdata.n_var == len(mask_var), sub_vdata.n_var

    assert sub_vdata.layers.shape == (1, 2, [5, 5], 4)
    assert sub_vdata.layers.shape == (1, 2, sub_vdata.n_obs, sub_vdata.n_var)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_sub_setting()
