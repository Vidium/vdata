from pathlib import Path

import pytest
from ch5mpy import File
from ch5mpy import H5Mode

from vdata.core.attribute_proxy.attribute import AttributeProxy, NONE_VALUE


@pytest.fixture
def attr():
    h5_file = File('attribute_proxy', H5Mode.WRITE_TRUNCATE)
    # write data to h5 file directly
    h5_file.attrs['type'] = 'test_type'
    h5_file.attrs['name'] = 'test_name'
    h5_file.attrs['none'] = NONE_VALUE

    attr = AttributeProxy(h5_file)

    yield attr

    h5_file.close()
    Path('attribute_proxy').unlink()
