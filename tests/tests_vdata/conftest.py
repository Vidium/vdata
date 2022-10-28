import numpy as np
import pandas as pd
import pytest
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
