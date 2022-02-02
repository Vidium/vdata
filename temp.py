# coding: utf-8
# Created on 02/02/2022 16:03
# Author : matteo

# ====================================================
# imports
import vdata
import anndata


# ====================================================
# code
adata = anndata.read('/home/matteo/git/Real_platform/storage/mbouvier/EI52/uploads/combinedData_filtered.h5')
adata.obs['TP'] = [int(e[1:]) for e in adata.obs.batch]

v = vdata.VData(adata, time_col_name='TP')
print(v)
