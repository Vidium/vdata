# coding: utf-8
# Created on 18/01/2021 15:48
# Author : matteo

# ====================================================
# imports
from vdata import setLoggingLevel

from .test_object_creation_from_AnnData import test_object_creation_from_AnnData
from .test_subsetting import test_sub_setting
from .test_write import test_write


# ====================================================
# code
setLoggingLevel('INFO')

test_object_creation_from_AnnData()
test_sub_setting()
test_write()
