# coding: utf-8
# Created on 1/7/21 5:05 PM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
from vdata import VData


# ====================================================
# code
source_vdata_path = "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"
adata = sc.read(source_vdata_path)
from vdata import TemporalDataFrame
VData(adata)