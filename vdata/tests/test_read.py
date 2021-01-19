# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
from vdata import setLoggingLevel, read, read_from_csv, read_from_dict
from .test_write import test_write


# ====================================================
# code
def test_read():
    # first write data
    # test_write()

    # then load data
    # load from .h5 file
    v = read("~/vdata.h5")
    print(v)

    # load from csv files
    v = read_from_csv("~/vdata")
    print(v)

    # load from a dictionary


if __name__ == "__main__":
    setLoggingLevel('DEBUG')

    test_read()
