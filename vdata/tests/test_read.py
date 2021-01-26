# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd

from vdata import setLoggingLevel, read, read_from_csv, read_from_dict
from vdata.tests.test_write import test_write


# ====================================================
# code
def test_read():
    # # first write data
    # # test_write()
    #
    # # then load data
    # # load from .h5 file
    # v = read("~/vdata.h5")
    # print(v)
    #
    # # load from csv files
    # v = read_from_csv("~/vdata")
    # print(v)
    #
    # # load from a dictionary
    # data = {
    #     'RNA': {
    #         '0h': np.zeros((7, 3)),
    #         '5h': np.ones((3, 3)),
    #         '10h': 2 * np.ones((10, 3))
    #     },
    #     'Protein': {
    #         '0h': 10 * np.ones((7, 3)),
    #         '5h': 20 * np.ones((3, 3)),
    #         '10h': 30 * np.ones((10, 3))
    #     }
    # }
    #
    # v = read_from_dict(data)
    # print(v)

    # load from a dictionary of insilico data
    data = {
        'RNA': {
            '0h': np.zeros((10, 3)),
            '5h': np.ones((10, 3)),
            '10h': 2 * np.ones((10, 3))
        },
        'Protein': {
            '0h': 10 * np.ones((10, 3)),
            '5h': 20 * np.ones((10, 3)),
            '10h': 30 * np.ones((10, 3))
        }
    }

    obs = pd.DataFrame({'id_cells': range(30)})

    v = read_from_dict(data, obs=obs)
    print(v)


if __name__ == "__main__":
    setLoggingLevel('DEBUG')

    test_read()
