import numpy as np
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile

from vdata.core.VData import vdata


@pytest.fixture
def VData():
    genes = list(map(lambda x: "g_" + str(x), range(50)))
    cells = list(map(lambda x: "c_" + str(x), range(300)))

    v = vdata.VData(data={'data': pd.DataFrame(np.array(range(300 * 50)).reshape((300, 50)),
                                               index=cells,
                                               columns=genes)},
                    obs=pd.DataFrame({'col1': range(300)}, index=cells),
                    var=pd.DataFrame({'col1': range(50)}, index=genes),
                    time_list=['0h' for _ in range(100)] + ['1h' for _ in range(100)] + ['2h' for _ in range(100)])

    return v


@pytest.fixture
def VData_uns(request):
    if hasattr(request, 'param'):
        which = request.param

    else:
        which = 'plain'

    timepoints = pd.DataFrame({"value": ["0h"]})
    var = pd.DataFrame({"gene_name": ["g1", "g2", "g3"]})
    obs = vdata.TemporalDataFrame({'data': np.random.randint(0, 20, 6),
                                   'data_bis': np.random.randint(0, 20, 6)},
                                  time_list=["0h", "0h", "0h", "0h", "0h", "0h"])
    uns = {"colors": np.array(['blue', 'red', 'yellow']),
           "date": '25/01/2021'}

    data = pd.DataFrame(np.array([[10, 11, 12],
                                 [20, 21, 22],
                                 [30, 31, 32],
                                 [40, 41, 42],
                                 [50, 51, 52],
                                 [60, 61, 62]]))

    v = vdata.VData(data, timepoints=timepoints, obs=obs, var=var, uns=uns, name=1)

    if 'backed' in which:
        tmp_file = NamedTemporaryFile(mode='w+b', suffix='.vd')

        v.write(tmp_file.name)
        yield v
        v.file.close()

    else:
        yield v
