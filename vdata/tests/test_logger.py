# coding: utf-8
# Created on 12/8/20 3:56 PM
# Author : matteo

# ====================================================
# imports
from .._IO.logger import generalLogger
generalLogger.set_level("WARNING")

# ====================================================
# code
a=1/0

generalLogger.error("test")
