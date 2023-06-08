from . import logger


# Errors
class VBaseError(Exception):
    """
    Base class for custom error. Error messages are redirected to the logger instead of being printed directly.
    """
    def __init__(self, msg: str = ""):
        self.msg = msg

    def __str__(self) -> str:
        logger.generalLogger.error(self.msg)
        return self.msg


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


class VLockError(VBaseError):
    """
    Custom error for tdf lock errors.
    """
    pass


class VClosedFileError(VBaseError):
    """
    Custom error for tdf lock errors.
    """
    def __init__(self, msg: str = ""):
        self.msg = "Closed backing file !"


class VReadOnlyError(VBaseError):
    """
    Custom error for modifications on read only data.
    """

    def __init__(self, msg: str = ""):
        self.msg = "Read-only file !"
        