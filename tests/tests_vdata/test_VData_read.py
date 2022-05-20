# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd
from time import perf_counter
from pathlib import Path

import vdata
from . import data
from .test_VData_write import out_test_VData_write


# ====================================================
# code
def test_VData_read():
    # first write data
    out_test_VData_write()

    output_dir = Path(__file__).parent.parent / 'ref'

    # then load data
    # load from .vd file
    v = vdata.read(output_dir / "vdata.vd")
    assert repr(v) == "Backed VData '1' with n_obs x n_var = [179, 24, 141, 256, 265, 238, 116, 149, 256, 293] " \
                      "x 1000 over 10 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'Time_hour', 'Cell_Type', 'Day'\n\t" \
                      "var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n\t" \
                      "timepoints: 'value'\n\t" \
                      "uns: 'colors', 'date'", repr(v)

    v.file.close()

    # load from csv files
    v = vdata.read_from_csv(output_dir / "vdata", name=2)
    assert repr(v) == "VData '2' with n_obs x n_var = [179, 24, 141, 256, 265, 238, 116, 149, 256, 293] x 1000 " \
                      "over 10 time points.\n\t" \
                      "layers: 'data'\n\t" \
                      "obs: 'Time_hour', 'Cell_Type', 'Day'\n\t" \
                      "var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n\t" \
                      "timepoints: 'value'", repr(v)

    # load from a dictionary
    obs = pd.DataFrame({'id_cells': range(20)})

    v = vdata.read_from_dict(data, obs=obs, name=3)
    assert repr(v) == "VData '3' with n_obs x n_var = [7, 3, 10] x 4 over 3 time points.\n\t" \
                      "layers: 'RNA', 'Protein'\n\t" \
                      "obs: 'id_cells'\n\t" \
                      "timepoints: 'value'", repr(v)


def test_VData_read_large():
    import cProfile

    output_dir = Path(__file__).parent.parent / 'ref'

    with cProfile.Profile() as prof:
        start = perf_counter()
        _ = vdata.read(output_dir / 'simulation.vd')
        elasped_time = perf_counter() - start

    prof.dump_stats('/home/matteo/Desktop/vdata.prof')

    assert elasped_time < 2


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_read()
