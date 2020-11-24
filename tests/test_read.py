# coding: utf-8
# Created on 11/23/20 9:45 AM
# Author : matteo

# ====================================================
# imports
from .._IO.read import read


# ====================================================
# code
vdata = read("/home/matteo/Desktop/vdata.h5", log_level='DEBUG')
print(vdata)
