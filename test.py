# coding: utf-8
# Created on 05/01/2022 10:13
# Author : matteo

# ====================================================
# imports
import numpy as np
import pandas as pd
from vdata.vdataframe import VDataFrame

from typing import Any, Union, Optional


# ====================================================
# code
def add_one(things: dict[Any, Union[VDataFrame, np.ndarray]]) -> None:
    ...


my_things: dict[Any, np.ndarray] = {'a': np.array([])}

add_one(my_things)
