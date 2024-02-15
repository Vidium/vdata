from tempfile import NamedTemporaryFile
from pathlib import Path

import vdata


def test_AnnData_conversion_to_VData(AnnData):
    with NamedTemporaryFile() as tmp_file:
        AnnData.write_h5ad(Path(tmp_file.name))

        vdata.convert_anndata_to_vdata(path=tmp_file.name, timepoints_column_name="Time_hour", inplace=False)

        with vdata.read(Path(tmp_file.name).with_suffix(".vd")) as v:
            repr_v = """Backed VData 'No_Name' ([3, 4, 3] obs x 3 vars over 3 time points).
	layers: 'X', 'data'
	obs: 'col1', 'Time_hour'
	timepoints: 'value'"""

            assert repr(v) == repr_v
