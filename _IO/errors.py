# coding: utf-8
# Created on 11/6/20 7:08 PM
# Author : matteo

# ====================================================
# imports
from .logger import generalLogger


# ====================================================
# code
# Errors
class VBaseError(Exception):
    """
    Base class for custom error. Error messages are redirected to the logger instead of being printed directly.
    """
    def __init__(self, msg: str = ""):
        self.msg = msg

    def __str__(self) -> str:
        generalLogger.error(self.msg)
        return ""


class VTypeError(VBaseError):
    """
    Custom error for type errors.
    """
    pass


class VValueError(VBaseError):
    """
    Custom error for value errors.
    """
    pass


class ShapeError(VBaseError):
    """
    Custom error for errors in variable shapes.
    """
    pass


class IncoherenceError(VBaseError):
    """
    Custom error for incoherent data formats.
    """
    pass


class VPathError(VBaseError):
    """
    Custom error for path errors.
    """
    pass


class VAttributeError(VBaseError):
    """
    Custom error for attribute errors.
    """
    pass
