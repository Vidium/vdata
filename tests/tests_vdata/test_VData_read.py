# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import pandas as pd

import vdata

from .test_VData_write import out_test_VData_write


# ====================================================
# code
def test_VData_read() -> None:
    # first write data
    out_test_VData_write()

    output_dir = Path(__file__).parent.parent / 'ref'

    # then load data
    # load from .vd file
    v = vdata.VData.read(output_dir / "vdata.vd", 'r+')
    assert repr(v) == "Backed VData '1' ([179, 24, 141, 256, 265, 238, 116, 149, 256, 293] obs " \
                      "x 1000 vars over 10 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'Time_hour', 'Cell_Type', 'Day'\n" \
                      "\tvar: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n" \
                      "\ttimepoints: 'value'\n" \
                      "\tuns: 'colors', 'date'"

    v.close()


def test_VData_read_csv() -> None:
    out_test_VData_write()

    output_dir = Path(__file__).parent.parent / 'ref'
    
    # load from csv files
    v = vdata.VData.read_from_csv(output_dir / "vdata", name='2')
    assert repr(v) == "VData '2' ([179, 24, 141, 256, 265, 238, 116, 149, 256, 293] obs x 1000 vars " \
                      "over 10 time points).\n" \
                      "\tlayers: 'data'\n" \
                      "\tobs: 'Time_hour', 'Cell_Type', 'Day'\n" \
                      "\tvar: 'ensembl ID', 'gene_short_name', 'pval', 'qval'\n" \
                      "\ttimepoints: 'value'"


def test_VData_read_nan() -> None:
    genes = list(map(lambda x: "g_" + str(x), range(5)))
    cells = list(map(lambda x: "c_" + str(x), range(10)))

    v = vdata.VData(data={'data': pd.DataFrame(np.arange(50).reshape((10, 5)),
                                               index=cells,
                                               columns=genes)},
                    obs=pd.DataFrame({'col1': range(10)}, index=cells),
                    var=pd.DataFrame({'col1': range(5)}, index=genes),
                    varm={'X1': pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, np.nan, 8, 9]},
                                             index=genes)},
                    time_list=['0h', '0h', '0h', '0h', '0h', '1h', '1h', '1h', '1h', '1h'])

    tmp_file = NamedTemporaryFile(mode='w+b', suffix='.vd')
    v.write(tmp_file.name)
    
    rv = vdata.VData.read(tmp_file.name)
    
    assert np.isnan(rv.varm['X1'].loc['g_2', 'b'])


# TODO : corrupted test data
def TODO_test_VData_read_large() -> None:
    # import cProfile

    output_dir = Path(__file__).parent.parent / 'ref'

    # with cProfile.Profile() as prof:
    start = perf_counter()
    _ = vdata.VData.read(output_dir / 'simulation.vd', 'r+')
    elasped_time = perf_counter() - start

    # prof.dump_stats('/home/matteo/Desktop/vdata.prof')

    assert elasped_time < 2
