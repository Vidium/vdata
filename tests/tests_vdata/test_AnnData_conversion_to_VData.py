# coding: utf-8
# Created on 17/03/2022 14:13
# Author : matteo

# ====================================================
# imports
import vdata

from pathlib import Path


# ====================================================
# code
def out_test_AnnData_conversion_to_VData():
    # TODO : repair conversion function
    input_dir = Path(__file__).parent.parent / 'ref'

    vdata.convert_anndata_to_vdata(file=input_dir / 'sel_JB_scRNAseq.h5ad',
                                   time_column_name='Time_hour',
                                   inplace=False)

    v = vdata.read(input_dir / 'sel_JB_scRNAseq.vd')

    print(v)


if __name__ == '__main__':
    vdata.setLoggingLevel('DEBUG')

    out_test_AnnData_conversion_to_VData()
