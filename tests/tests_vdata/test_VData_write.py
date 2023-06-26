from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import scanpy as sc
from ch5mpy import H5Array, H5Dict

import vdata


@pytest.fixture
def vdata_from_anndata() -> vdata.VData:
    adata = sc.read(Path(__file__).parent.parent / 'ref' / 'sel_JB_scRNAseq.h5ad')
    adata.obs['tp'] = adata.obs.Time_hour.astype(str) + 'h'

    v = vdata.VData(adata, time_col_name='tp', name='1')
    v.uns = {"colors": ['blue', 'red', 'yellow'],
             "date": '25/01/2021'}
    
    return v


def test_VData_can_write_in_h5(vdata_from_anndata) -> None:
    with NamedTemporaryFile(mode='a', suffix='.vd') as file:
        vdata_from_anndata.write(file.name)


def test_VData_can_write_in_csv(vdata_from_anndata) -> None:
    with TemporaryDirectory() as dir:
        vdata_from_anndata.write_to_csv(dir)


def test_VData_view_can_write_in_h5(vdata_from_anndata) -> None:
    list_cells = ['plate2_H12_G01_G10', 'plate1_A09_A09_A12', 'plate2_H12_D01_H10', 'plate1_A01_A01_B01']
    list_genes = ['ENSG00000260919.1', 'ENSG00000267586.6', 'ENSG00000268595.1', 'ENSG00000255794.7']

    sub_v = vdata_from_anndata[:, list_cells, list_genes]

    with NamedTemporaryFile(mode='a', suffix='.vd') as file:
        sub_v.write(file.name)


def test_VData_view_can_write_in_csv(vdata_from_anndata) -> None:
    list_cells = ['plate2_H12_G01_G10', 'plate1_A09_A09_A12', 'plate2_H12_D01_H10', 'plate1_A01_A01_B01']
    list_genes = ['ENSG00000260919.1', 'ENSG00000267586.6', 'ENSG00000268595.1', 'ENSG00000255794.7']

    sub_v = vdata_from_anndata[:, list_cells, list_genes]
    
    with TemporaryDirectory() as dir:
        sub_v.write_to_csv(dir)


def test_VData_write_should_convert_uns_to_BackedDict() -> None:
    v = vdata.VData(uns={'test': np.array([1, 2, 3])})

    with NamedTemporaryFile(mode='w+b', suffix='.vd') as tmp_file:
        v.write(tmp_file.name)
        assert isinstance(v.uns, H5Dict)


def test_VData_write_should_convert_uns_arrays_to_datasetProxies() -> None:
    v = vdata.VData(uns={'test': np.array([1, 2, 3])})

    with NamedTemporaryFile(mode='w+b', suffix='.vd') as tmp_file:
        v.write(tmp_file.name)
        assert isinstance(v.uns['test'], H5Array)


def test_backed_VData_new_layer_should_be_backed(backed_VData: vdata.VData) -> None:
    backed_VData.layers['data_copy'] = backed_VData.layers['data'].copy()
    backed_VData.layers['data_copy'][:] = 100

    assert backed_VData.layers['data_copy'].is_backed
