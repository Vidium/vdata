# coding: utf-8
# Created on 1/7/21 11:30 AM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
import vdata
from vdata import VData

vdata.setLoggingLevel('DEBUG')

# ====================================================
# code
source_vdata_path = "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

adata = sc.read(source_vdata_path)

print(adata)

vdata = VData(adata)

print(vdata)
print(vdata.var)

print(vdata[0, :, 'ENSG00000255794.7'])
