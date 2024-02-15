from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

import vdata


def test_VData_read() -> None:
    v = vdata.read(Path(__file__).parent.parent / "ref" / "vdata.vd", vdata.H5Mode.READ)
    v_repr = """Backed VData 'ref' ([10, 10, 10] obs x 3 vars over 3 time points).
	layers: 'data'
	obs: 'col1', 'col2'
	var: 'gene_name'
	timepoints: 'value'
	uns: 'colors', 'date'"""

    assert repr(v) == v_repr

    v.close()


def test_VData_read_csv() -> None:
    v = vdata.read_from_csv(Path(__file__).parent.parent / "ref" / "vdata", name="2")
    v_repr = """VData '2' ([10, 10, 10] obs x 3 vars over 3 time points).
	layers: 'data'
	obs: 'col1', 'col2'
	var: 'gene_name'
	timepoints: 'value'"""

    assert repr(v) == v_repr


def test_VData_read_nan() -> None:
    genes = list(map(lambda x: "g_" + str(x), range(5)))
    cells = list(map(lambda x: "c_" + str(x), range(10)))

    v = vdata.VData(
        data={"data": pd.DataFrame(np.arange(50).reshape((10, 5)), index=cells, columns=genes)},
        obs=pd.DataFrame({"col1": range(10)}, index=cells),
        var=pd.DataFrame({"col1": range(5)}, index=genes),
        varm={"X1": pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, np.nan, 8, 9]}, index=genes)},
        timepoints_list=["0h", "0h", "0h", "0h", "0h", "1h", "1h", "1h", "1h", "1h"],
    )

    with NamedTemporaryFile(mode="w+b", suffix=".vd") as tmp_file:
        v.write(tmp_file.name)

        rv = vdata.VData.read(tmp_file.name)

        assert np.isnan(rv.varm["X1"].loc["g_2", "b"])
