from pathlib import Path
from tempfile import NamedTemporaryFile

import ch5mpy as ch
import pytest

from vdata import TemporalDataFrame
from vdata.tdf import TemporalDataFrameBase


# ====================================================
# code
@pytest.mark.parametrize("TDF", ["plain", "view", "backed", "backed view"], indirect=True)
def test_can_read_written_TDF(TDF: TemporalDataFrameBase) -> None:
    with NamedTemporaryFile() as tmp_file:
        TDF.write(tmp_file.name)

        tdf = TemporalDataFrame.read(tmp_file.name)
        assert isinstance(tdf, TemporalDataFrameBase)

        tdf.close()


@pytest.mark.usefixtures("h5_file")
class TestWriteTDF:
    h5_file: ch.File

    def test_wrote_correct_attributes(self) -> None:
        assert sorted(list(self.h5_file.attrs.keys())) == [
            "__h5_class__",
            "__h5_type__",
            "locked_columns",
            "locked_indices",
            "name",
            "repeating_index",
            "timepoints_column_name",
        ]

    def test_wrote_datasets(self) -> None:
        assert sorted(list(self.h5_file.keys())) == [
            "columns_numerical",
            "columns_string",
            "index",
            "numerical_array",
            "string_array",
            "timepoints_index",
        ]

    @pytest.mark.parametrize("TDF", ["plain"], indirect=True)
    def test_wrote_all_data_correctly(self, TDF: TemporalDataFrame) -> None:
        written_tdf = TemporalDataFrame.read(self.h5_file)

        assert written_tdf == TDF
