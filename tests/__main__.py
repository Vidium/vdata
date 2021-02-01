# coding: utf-8
# Created on 18/01/2021 15:48
# Author : matteo

# ====================================================
# imports
import vdata

from tests.tests_TDF.test_TDF_creation import test_TDF_creation
from tests.tests_TDF.test_TDF_subsetting import test_TDF_sub_setting

from tests.tests_vdata.test_VData_creation import test_VData_creation, test_VData_creation_on_dtype, \
    test_VData_creation_with_uns
from tests.tests_vdata.test_VData_creation_from_AnnData import test_VData_creation_from_AnnData
from tests.tests_vdata.test_VData_subsetting import test_VData_sub_setting
from tests.tests_vdata.test_VData_conversion_to_AnnData import test_VData_conversion_to_AnnData
from tests.tests_vdata.test_VData_read import test_VData_read


# ====================================================
# code
vdata.setLoggingLevel('INFO')

# TDF creation
print('\n >>> TemporalDataFrame creation <<< \n')
test_TDF_creation()

# TDF seb-setting
print('\n >>> TemporalDataFrame sub-setting <<< \n')
test_TDF_sub_setting()

# VData creation
print('\n >>> VData creation <<< \n')
test_VData_creation()
test_VData_creation_on_dtype()
test_VData_creation_with_uns()

print('\n >>> VData creation from AnnData <<< \n')
test_VData_creation_from_AnnData()

# VData sub-setting
print('\n >>> VData sub-setting <<< \n')
test_VData_sub_setting()

# VData conversion
print('\n >>> VData conversion to AnnData <<< \n')
test_VData_conversion_to_AnnData()

# VData read / write
print('\n >>> VData read / write <<< \n')
test_VData_read()
