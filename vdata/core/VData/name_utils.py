# coding: utf-8
# Created on 08/03/2021 19:29
# Author : matteo

# ====================================================
# imports
import pandas as pd
from typing import Union

from vdata.core.tdf import TemporalDataFrame
from vdata.vdataframe import VDataFrame

# ====================================================
# code
DataFrame = Union[pd.DataFrame, TemporalDataFrame, VDataFrame]
