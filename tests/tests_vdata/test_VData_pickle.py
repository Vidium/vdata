import pytest
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

import ch5mpy as ch
import vdata


REF_DIR = Path(__file__).parent.parent / "ref"


def VData_pickle_dump(pkl_file):
    with vdata.read(REF_DIR / "vdata.vd", mode=ch.H5Mode.READ_WRITE) as v1:
        # FIXME: pickling H5Arrays does not work for now
        # obj = v1.uns.colors
        # pickle.dump(obj, save_file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(v1, pkl_file, pickle.HIGHEST_PROTOCOL)


@pytest.mark.xfail
def test_VData_pickle_load():
    with NamedTemporaryFile(mode="w+b") as pkl_file:
        VData_pickle_dump(pkl_file)

        with vdata.read_from_pickle(pkl_file) as v2:
            repr_v2 = """Backed VData '1' with n_obs x n_var = [179, 24, 141, 256, 265, 238, 116, 149, 256, 293] x 1000 over 10 time points.
    layers: 'data'
    obs: 'Time_hour', 'Cell_Type', 'Day'
    var: 'ensembl ID', 'gene_short_name', 'pval', 'qval'
    timepoints: 'value'
    uns: 'colors', 'date'"""
            assert repr(v2) == repr_v2
