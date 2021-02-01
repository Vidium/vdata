# coding: utf-8
# Created on 1/7/21 11:30 AM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
import vdata


# ====================================================
# code
def test_VData_creation_from_AnnData():
    source_vdata_path = \
        "/home/matteo/Desktop/JN/Project/DMD/2-Genetic_Dynamic_Characterization/1-Dynamic_analysis/sel_JB_scRNAseq.h5ad"

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col='Time_hour')

    assert repr(v) == "Vdata object with n_obs x n_var = [179, 24, 141, 256, 265, 238, 116, 149, 256, 293] x 1000 " \
                      "over 10 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'Cell_Type', 'Day', 'Time_hour'\n\t" \
                      "var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n\t" \
                      "time_points: 'value'", repr(v)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_creation_from_AnnData()

