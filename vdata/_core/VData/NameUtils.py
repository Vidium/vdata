# coding: utf-8
# Created on 08/03/2021 19:29
# Author : matteo

# ====================================================
# imports
import pandas as pd
from typing import Union

from ..TDF import TemporalDataFrame
from vdata.VDataFrame import VDataFrame

# ====================================================
# code
DataFrame = Union[pd.DataFrame, TemporalDataFrame, VDataFrame]
