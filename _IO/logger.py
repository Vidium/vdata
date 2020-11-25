# coding: utf-8
# Created on 11/6/20 5:55 PM
# Author : matteo

# ====================================================
# imports
import os
import sys
import logging.config
import inspect
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

from . import errors
from ..NameUtils import LoggingLevel


# ====================================================
class Tb:
    trace: Optional[TracebackType] = None
    exception: Type[BaseException] = BaseException


# code
class _VLogger:
    """
    Custom logger for reporting messages to the console.
    Logging levels are :
        - DEBUG
        - INFO
        - WARNING
        - ERROR
        - CRITICAL

    The default minimal level for logging is <INFO>.
    """

    def __init__(self, logger_level: LoggingLevel = "WARNING"):
        """
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # load configuration from logging.conf
        logging.config.fileConfig(Path(os.path.dirname(__file__)) / "logger.conf", defaults={'log_level': logger_level})

        # get logger
        self.logger = logging.getLogger('root.vlogger')

    def set_level(self, logger_level: LoggingLevel) -> None:
        """
        Re-init the logger, for setting new minimal logging level
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger.setLevel(logger_level)
        for handler in self.logger.handlers:
            handler.setLevel(logger_level)

    @staticmethod
    def _getBaseMsg(msg: str) -> str:
        """
        Build the message to log with format <[fileName.py] msg>

        :param msg: the message to be logged
        :return: the formatted message
        """
        # TODO : add error type to the message [file.py - VValueError]

        # Get the name of the file that called the logger for displaying where the message came from
        if Tb.trace is None:
            frames = inspect.stack(0)

            caller_filename = frames[0].filename
            index = 0

            while index < len(frames) - 1 and (caller_filename.endswith("logger.py") or caller_filename.endswith("errors.py")):
                index += 1
                caller_filename = frames[index].filename

            caller = os.path.splitext(os.path.basename(caller_filename))[0]

            # return base message
            return f"[{caller}.py] {msg}"

        else:
            caller = ""

            while Tb.trace is not None:
                caller = Tb.trace.tb_frame.f_code.co_filename
                Tb.trace = Tb.trace.tb_next

            return f"[{os.path.basename(caller)} : {Tb.exception.__name__}] {msg}"

    def debug(self, msg: str) -> None:
        """
        Log a debug message (level 10)

        :param msg: the message to be logged
        """
        self.logger.debug(self._getBaseMsg(msg))

    def info(self, msg: str) -> None:
        """
        Log an info message (level 20)

        :param msg: the message to be logged
        """
        self.logger.info(self._getBaseMsg(msg))

    def warning(self, msg: str) -> None:
        """
        Log a warning message (level 30)

        :param msg: the message to be logged
        """
        self.logger.warning(self._getBaseMsg(msg))

    def error(self, msg: str) -> None:
        """
        Log an error message (level 40)

        :param msg: the message to be logged
        """
        self.logger.error(self._getBaseMsg(msg))
        quit()

    def uncaught_error(self, msg: str) -> None:
        """
        Log and uncaught (not originating from a custom error class) error message (level 40)

        :param msg: the message to be logged
        """
        last = None
        while Tb.trace is not None:
            last = Tb.trace.tb_frame
            Tb.trace = Tb.trace.tb_next

        # last.f_globals['__package__']
        self.logger.error(f"[{last.f_globals['__name__'] if last is not None else 'UNCAUGHT'} : {Tb.exception.__name__}] {msg}")

    def critical(self, msg: str) -> None:
        """
        Log a critical message (level 50)

        :param msg: the message to be logged
        """
        self.logger.critical(self._getBaseMsg(msg))


generalLogger = _VLogger()


# basic definition of the excepthook for proper behavior of the logger
# this will be updated by vdata.py as soon as the user creates a VData (and sets a logger level)
def exception_handler(exception_type, exception, traceback):
    Tb.trace = traceback
    Tb.exception = exception_type

    if not issubclass(exception_type, errors.VBaseError):
        generalLogger.uncaught_error(exception)
    else:
        print(exception)


original_excepthook = sys.excepthook
sys.excepthook = exception_handler
