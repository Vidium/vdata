# coding: utf-8
# Created on 11/6/20 7:08 PM
# Author : matteo

# ====================================================
# imports
from .logger import generalLogger


# ====================================================
# code
# Errors
class BaseError(Exception):
    def __init__(self, msg: str = ""):
        self.msg = msg

    def __str__(self) -> None:
        generalLogger.error(self.msg)


class VTypeError(BaseError):
    pass


class VValueError(BaseError):
    pass


class ShapeError(BaseError):
    pass


class NotEnoughDataError(BaseError):
    pass


class IncoherenceError(BaseError):
    pass
