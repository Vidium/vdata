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
    """
    Base class for custom error. Error messages are redirected to the logger instead of being printed directly.
    """
    def __init__(self, msg: str = ""):
        self.msg = msg

    def __str__(self) -> str:
        generalLogger.error(self.msg)
        return ""


class VTypeError(BaseError):
    """
    Custom error for type errors.
    """
    pass


class VValueError(BaseError):
    """
    Custom error for value errors.
    """
    pass


class ShapeError(BaseError):
    """
    Custom error for errors in variable shapes.
    """
    pass


# todo : might not be usefull, check in vdata
class NotEnoughDataError(BaseError):
    """
    Custom errors for missing data.
    """
    pass


class IncoherenceError(BaseError):
    """
    Custom error for incoherent data formats.
    """
    pass
