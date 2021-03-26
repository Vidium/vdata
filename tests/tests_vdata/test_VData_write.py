# coding: utf-8
# Created on 11/20/20 11:04 AM
# Author : matteo

# ====================================================
# imports
import os
import shutil
import scanpy as sc
from pathlib import Path

import vdata


# ====================================================
# code
def out_test_VData_write():
    """
    This test is has the 'out_' prefix to exclude it from pytest since it is called by test_VData_read.
    """
    output_dir = Path(__file__).parent.parent / 'ref'

    if os.path.exists(output_dir / 'vdata'):
        shutil.rmtree(output_dir / 'vdata')

    # create vdata
    source_vdata_path = output_dir / 'sel_JB_scRNAseq'

    adata = sc.read(source_vdata_path)

    v = vdata.VData(adata, time_col_name='Time_hour')

    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    v.uns = uns

    # write vdata in h5 file format
    v.write(output_dir / "vdata.h5")

    # write vdata in csv files
    v.write_to_csv(output_dir / "vdata")


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    out_test_VData_write()
