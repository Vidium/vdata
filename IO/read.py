# coding: utf-8
# Created on 11/12/20 4:51 PM
# Author : matteo

# ====================================================
# imports
import os
import pickle
from pathlib import Path
from typing import Union

from .errors import VValueError
from ..core.vdata import VData


# ====================================================
# code
def read(file: Union[Path, str]) -> VData:
    """
    Load a pickled VData object.
    Example :
    >>> import vdata
    >>> vdata.read("/path/to/file.p")

    :param file: path to a saved VData object
    """
    # make sure file is a path
    if not isinstance(file, Path):
        file = Path(file)

    # make sure the path exists
    if not os.path.exists(file):
        raise VValueError(f"The path {file} does not exist.")

    with open(file, 'rb') as save_file:
        vdata = pickle.load(save_file)

    return vdata
