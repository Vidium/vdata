import numpy as np
import pandas as pd

import vdata


# ====================================================
# code
def test_VData_modification(backed_VData: vdata.VData) -> None:
    # set once
    backed_VData.obsm["X"] = vdata.TemporalDataFrame(
        data=pd.DataFrame({"col1": range(backed_VData.n_obs_total)}),
        timepoints=backed_VData.obs.timepoints_column,
        index=backed_VData.obs.index,
    )

    assert "X" in backed_VData.obsm.keys()

    # set a second time
    backed_VData.obsm["X"] = vdata.TemporalDataFrame(
        data=pd.DataFrame({"col1": 2 * np.arange(backed_VData.n_obs_total)}),
        timepoints=backed_VData.obs.timepoints_column,
        index=backed_VData.obs.index,
    )

    assert "X" in backed_VData.obsm.keys()
    del backed_VData.obsm["X"]
    assert "X" not in backed_VData.obsm.keys()

    backed_VData.close()


def test_VData_set_index(VData: vdata.VData) -> None:
    # not repeating ==> not repeating
    VData.set_obs_index(values=range(VData.n_obs_total))

    assert np.array_equal(VData.obs.index, range(VData.n_obs_total))
    assert np.array_equal(VData.layers["data"].index, range(VData.n_obs_total))


def test_VData_set_index_repeating(VData: vdata.VData) -> None:
    # not repeating ==> repeating
    new_index = vdata.Index(range(100), repeats=VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)

    # repeating ==> repeating
    new_index = vdata.Index(range(0, 200, 2), repeats=VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)

    # repeating ==> not repeating
    new_index = np.arange(100 * VData.n_timepoints)

    VData.set_obs_index(values=new_index)

    assert np.array_equal(VData.obs.index, new_index)
    assert np.array_equal(VData.layers["data"].index, new_index)


def test_VData_set_new_obsm_is_written_to_file(backed_VData: vdata.VData) -> None:
    cells = list(map(lambda x: "c_" + str(x), range(300)))

    backed_VData.obsm["pca"] = vdata.TemporalDataFrame(
        np.ones((300, 2)),
        index=cells,
        columns=["X1", "X2"],
        timepoints=["0h" for _ in range(100)] + ["1h" for _ in range(100)] + ["2h" for _ in range(100)],
    )

    assert "pca" in backed_VData.obsm.keys()
    assert "pca" in backed_VData.data.obsm.keys()
