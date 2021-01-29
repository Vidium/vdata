# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
import pandas as pd

import vdata
from . import data
from .test_VData_write import test_VData_write


# ====================================================
# code
def test_VData_read():
    # first write data
    test_VData_write()

    # then load data
    # load from .h5 file
    v = vdata.read("~/vdata.h5")
    print(v)

    # load from csv files
    v = vdata.read_from_csv("~/vdata")
    print(v)

    # load from a dictionary
    obs = pd.DataFrame({'id_cells': range(30)})

    v = vdata.read_from_dict(data, obs=obs)
    print(v)


if __name__ == "__main__":
    vdata.setLoggingLevel('DEBUG')

    test_VData_read()
