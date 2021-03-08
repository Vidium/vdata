# coding: utf-8
# Created on 11/20/20 11:04 AM
# Author : matteo

# ====================================================
# imports
import os
import scanpy as sc

import vdata


# ====================================================
# code
def test_VData_write():
    os.system('rm -rf ~/vdata')

    # create vdata
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col_name='Time_hour')

    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    v.uns = uns

    # print(v)

    # write vdata in h5 file format
    v.write("~/vdata.h5")

    print("------------------------------------------")

    # write vdata in csv files
    v.write_to_csv("~/vdata")


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_write()
