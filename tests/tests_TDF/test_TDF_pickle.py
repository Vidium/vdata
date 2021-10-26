# coding: utf-8
# Created on 18/10/2021 16:01
# Author : matteo

# ====================================================
# imports
import pickle
from pathlib import Path

import vdata


# ====================================================
# code
REF_DIR = Path(__file__).parent.parent / 'ref'


def test_TDF_pickle_dump():
    _TDF = vdata.TemporalDataFrame({'col1': [1, 2, 3, 4, 5, 6],
                                    'col2': [7, 8, 9, 10, 11, 12]},
                                   name='pickleable TDF',
                                   index=[f"C_{i}" for i in range(6)],
                                   time_list=['0h', '0h', '0h', '0h', '1h', '1h'])

    _TDF.write(REF_DIR / 'pickled_TDF.h5')
    _TDF = vdata.read_TemporalDataFrame(REF_DIR / 'pickled_TDF.h5')

    p = str(REF_DIR / 'pickled_TDF.pkl')

    pickle.dump(_TDF, open(p, 'wb'))


def test_TDF_pickle_load():
    p = str(REF_DIR / 'pickled_TDF.pkl')

    _TDF = pickle.load(open(p, 'rb'))

    assert repr(_TDF) == "Backed TemporalDataFrame 'pickleable TDF'\n" \
                         "\033[4mTime point : 0.0 hours\033[0m\n" \
                         "     col1  col2\n" \
                         "C_0     1     7\n" \
                         "C_1     2     8\n" \
                         "C_2     3     9\n" \
                         "C_3     4    10\n" \
                         "\n" \
                         "[4 x 2]\n" \
                         "\n" \
                         "\033[4mTime point : 1.0 hours\033[0m\n" \
                         "     col1  col2\n" \
                         "C_4     5    11\n" \
                         "C_5     6    12\n" \
                         "\n" \
                         "[2 x 2]\n\n", repr(_TDF)

    _TDF.file.file.close()


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_TDF_pickle_dump()
    test_TDF_pickle_load()
