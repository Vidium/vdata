# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
from .._IO.read import read_from_csv
from .._IO.errors import VTypeError


# ====================================================
# code
vdata = read_from_csv("/home/matteo/Desktop/vdata", log_level='DEBUG')
raise ValueError('test')
print(vdata)
