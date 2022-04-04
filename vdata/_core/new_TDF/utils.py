# coding: utf-8
# Created on 01/04/2022 08:52
# Author : matteo

# ====================================================
# imports
from collections import Collection

from typing import Any


# ====================================================
# code
def is_collection(obj: Any) -> bool:
    return isinstance(obj, Collection) and not isinstance(obj, (str, bytes, bytearray))
