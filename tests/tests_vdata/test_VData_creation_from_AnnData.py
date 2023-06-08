# coding: utf-8
# Created on 1/7/21 11:30 AM
# Author : matteo

# ====================================================
# imports
from pathlib import Path

import scanpy as sc

import vdata


# ====================================================
# code
def test_VData_creation_from_AnnData() -> None:
    output_dir = Path(__file__).parent.parent / 'ref'

    adata = sc.read(output_dir / 'sel_JB_scRNAseq.h5ad')

    adata.obs['tp'] = adata.obs.Time_hour.astype(str) + 'h'

    v = vdata.VData(adata, time_col_name='tp')

    assert repr(v) == "VData 'No_Name' ([179, 24, 141, 256, 265, 238, 116, 149, 256, 293] obs x 1000 vars " \
                      "over 10 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'Time_hour', 'Cell_Type', 'Day'\n" \
                      "\tvar: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n" \
                      "\ttimepoints: 'value'"
