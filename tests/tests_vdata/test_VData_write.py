from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from ch5mpy import H5Array, H5Dict

import vdata


def test_VData_can_write_in_h5(VData_uns) -> None:
    with NamedTemporaryFile(mode="a", suffix=".vd") as file:
        VData_uns.write(file.name)


def test_VData_can_write_in_csv(VData_uns) -> None:
    with TemporaryDirectory() as dir:
        VData_uns.write_to_csv(dir)


def test_VData_view_can_write_in_h5(VData_uns) -> None:
    list_cells = [3, 1, 4, 2]
    list_genes = ["g3", "g1"]

    sub_v = VData_uns[:, list_cells, list_genes]

    with NamedTemporaryFile(mode="a", suffix=".vd") as file:
        sub_v.write(file.name)


def test_VData_view_can_write_in_csv(VData_uns) -> None:
    list_cells = [3, 1, 4, 2]
    list_genes = ["g3", "g1"]

    sub_v = VData_uns[:, list_cells, list_genes]

    with TemporaryDirectory() as dir:
        sub_v.write_to_csv(dir)


def test_VData_write_should_convert_uns_to_BackedDict() -> None:
    v = vdata.VData(uns={"test": np.array([1, 2, 3])})

    with NamedTemporaryFile(mode="w+b", suffix=".vd") as tmp_file:
        v.write(tmp_file.name)
        assert isinstance(v.uns, H5Dict)


def test_VData_write_should_convert_uns_arrays_to_datasetProxies() -> None:
    v = vdata.VData(uns={"test": np.array([1, 2, 3])})

    with NamedTemporaryFile(mode="w+b", suffix=".vd") as tmp_file:
        v.write(tmp_file.name)
        assert isinstance(v.uns["test"], H5Array)


def test_backed_VData_new_layer_should_be_backed(backed_VData: vdata.VData) -> None:
    backed_VData.layers["data_copy"] = backed_VData.layers["data"].copy()
    backed_VData.layers["data_copy"][:] = 100

    assert backed_VData.layers["data_copy"].is_backed
