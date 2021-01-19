# coding: utf-8
# Created on 1/7/21 11:30 AM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
import vdata
from vdata import VData


# ====================================================
# code
def test_object_creation_from_AnnData():
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    print(adata)

    v = VData(adata, time_col='Time_hour')

    print(v)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_object_creation_from_AnnData()

