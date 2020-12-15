# coding: utf-8
# Created on 12/8/20 3:56 PM
# Author : matteo

# ====================================================
# imports
from .._IO.logger import generalLogger
from .._IO.errors import VValueError

generalLogger.level = "DEBUG"

# ====================================================
# code
# a=1/0

generalLogger.warning("hello")
generalLogger.info('world')

# generalLogger.error("test")

raise VValueError("second test")
