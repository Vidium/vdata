# coding: utf-8
# Created on 19/10/2021 11:53
# Author : matteo

# ====================================================
# imports
import pickle
from pathlib import Path

import vdata


# ====================================================
# code
REF_DIR = Path(__file__).parent.parent / 'ref'


def VData_pickle_dump():
    v1 = vdata.read(REF_DIR / "vdata.vd")

    if (REF_DIR / 'pickled_vdata.pkl').exists():
        (REF_DIR / 'pickled_vdata.pkl').unlink()

    with open(REF_DIR / 'pickled_vdata.pkl', 'wb') as save_file:
        pickle.dump(v1, save_file, pickle.HIGHEST_PROTOCOL)

    v1.file.close()


def test_VData_pickle_load():
    VData_pickle_dump()

    v2 = pickle.load(open(REF_DIR / 'pickled_vdata.pkl', 'rb'))

    assert repr(v2) == "Backed VData '1' with n_obs x n_var = " \
                       "[179, 24, 141, 256, 265, 238, 116, 149, 256, 293] x 1000 over 10 time points.\n" \
                       "\tlayers: 'data'\n" \
                       "\tobs: 'Time_hour', 'Cell_Type', 'Day'\n" \
                       "\tvar: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n" \
                       "\ttimepoints: 'value'\n" \
                       "\tuns: 'colors', 'date'", repr(v2)

    v2.file.close()

    (REF_DIR / 'pickled_vdata.pkl').unlink()


if __name__ == '__main__':
    vdata.setLoggingLevel('INFO')

    test_VData_pickle_load()
