# coding: utf-8
# Created on 1/7/21 11:30 AM
# Author : matteo

# ====================================================
# imports
import scanpy as sc
from pathlib import Path

import vdata


# ====================================================
# code
def test_VData_creation_from_AnnData():
    output_dir = Path(__file__).parent.parent / 'ref'

    adata = sc.read(output_dir / 'sel_JB_scRNAseq.h5ad')

    adata.obs['tp'] = adata.obs.Time_hour.astype(str) + 'h'

    v = vdata.VData(adata, time_col_name='tp')

    assert repr(v) == "VData 'No_Name' with n_obs x n_var = [179, 24, 141, 256, 265, 238, 116, 149, 256, 293] x 1000 " \
                      "over 10 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'Time_hour', 'Cell_Type', 'Day'\n\t" \
                      "var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n\t" \
                      "timepoints: 'value'", repr(v)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_creation_from_AnnData()

