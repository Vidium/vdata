import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import vdata


def get_vdata():
    data = vdata.TemporalDataFrame(
        np.arange(30 * 3).reshape((30, 3)),
        timepoints=np.r_[np.repeat("0h", 10), np.repeat("1h", 10), np.repeat("2h", 10)],
    )
    timepoints = pd.DataFrame({"value": ["0h", "1h", "2h"]})
    obs = vdata.TemporalDataFrame(
        {"col1": np.arange(30), "col2": np.arange(30) * 10},
        timepoints=np.r_[np.repeat("0h", 10), np.repeat("1h", 10), np.repeat("2h", 10)],
    )
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    uns = {"colors": ["blue", "red", "yellow"], "date": "25/01/2024"}

    v = vdata.VData(data, timepoints=timepoints, obs=obs, var=var, uns=uns, name="ref")
    return v


def generate_vdata() -> None:
    v = get_vdata()

    path = Path(__file__).parent / "ref" / "vdata.vd"
    if path.exists():
        path.unlink()

    v.write(path)


def generate_vdata_csv() -> None:
    v = get_vdata()

    path = Path(__file__).parent / "ref" / "vdata/"
    if path.exists():
        shutil.rmtree(path)

    v.write_to_csv(path)


if __name__ == "__main__":
    generate_vdata()
    generate_vdata_csv()
