# coding: utf-8
# Created on 11/20/20 11:04 AM
# Author : matteo

# ====================================================
# imports
import os
import shutil
import numpy as np
import scanpy as sc
from pathlib import Path
from ch5mpy import H5Dict
from ch5mpy import H5Array
from tempfile import NamedTemporaryFile

import vdata
from vdata import VData


# ====================================================
# code
def out_test_VData_write():
    """
    This test has the 'out_' prefix to exclude it from pytest since it is called by test_VData_read.
    """
    output_dir = Path(__file__).parent.parent / 'ref'

    if os.path.exists(output_dir / 'vdata'):
        shutil.rmtree(output_dir / 'vdata')

    # create vdata
    source_vdata_path = output_dir / 'sel_JB_scRNAseq'

    adata = sc.read(source_vdata_path)

    adata.obs['tp'] = adata.obs.Time_hour.astype(str) + 'h'

    v = vdata.VData(adata, time_col_name='tp', name='1')

    uns = {"colors": ['blue', 'red', 'yellow'],
           "date": '25/01/2021'}

    v.uns = uns

    # write vdata in h5 file format
    v.write(output_dir / "vdata.vd")

    # write vdata in csv files
    v.write_to_csv(output_dir / "vdata")

    v.file.close()


def test_VData_view_write():
    output_dir = Path(__file__).parent.parent / 'ref'

    if os.path.exists(output_dir / 'sub_vdata'):
        shutil.rmtree(output_dir / 'sub_vdata')

    # create vdata
    source_vdata_path = output_dir / 'sel_JB_scRNAseq'

    adata = sc.read(source_vdata_path)

    adata.obs['tp'] = adata.obs.Time_hour.astype(str) + 'h'

    v = vdata.VData(adata, time_col_name='tp')

    list_cells = ['plate2_H12_G01_G10', 'plate1_A09_A09_A12', 'plate2_H12_D01_H10', 'plate1_A01_A01_B01']
    list_genes = ['ENSG00000260919.1', 'ENSG00000267586.6', 'ENSG00000268595.1', 'ENSG00000255794.7']

    sub_v = v[:, list_cells, list_genes]

    # write vdata in h5 file format
    sub_v.write(output_dir / "sub_vdata.vd")

    # write vdata in csv files
    sub_v.write_to_csv(output_dir / "sub_vdata")


def test_VData_write_should_convert_uns_to_BackedDict():
    v = VData(uns={'test': np.array([1, 2, 3])})

    tmp_file = NamedTemporaryFile(mode='w+b', suffix='.vd')
    v.write(tmp_file.name)

    assert isinstance(v.uns, H5Dict)


def test_VData_write_should_convert_uns_arrays_to_datasetProxies():
    v = VData(uns={'test': np.array([1, 2, 3])})

    tmp_file = NamedTemporaryFile(mode='w+b', suffix='.vd')
    v.write(tmp_file.name)

    assert isinstance(v.uns['test'], H5Array)


def test_backed_VData_new_layer_should_be_backed():
    output_dir = Path(__file__).parent.parent / 'ref'
    VData = vdata.read(output_dir / "vdata.vd", 'r+')

    VData.layers['data_copy'] = VData.layers['data'].copy()
    VData.layers['data_copy'][:] = 100

    assert VData.layers['data_copy'].is_backed
