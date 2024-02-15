import vdata


def test_VData_creation_from_AnnData(AnnData) -> None:
    v = vdata.VData(AnnData, time_col_name="Time_hour")
    v_repr = """VData 'No_Name' ([3, 4, 3] obs x 3 vars over 3 time points).
	layers: 'data'
	obs: 'col1'
	timepoints: 'value'"""

    assert repr(v) == v_repr
