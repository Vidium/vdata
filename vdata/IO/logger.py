# coding: utf-8
# Created on 11/6/20 5:55 PM
# Author : matteo

# ====================================================
# imports
import os
import sys
import logging.config
import traceback
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

from . import errors
from ..name_utils import LoggingLevel, LoggingLevels

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ====================================================
colors = {"TCYAN": '\033[36m', "TORANGE": '\033[33m', "TRED": '\033[31m',
          "BBLACK": '\033[40m', "BGREY": '\033[100m',
          "ENDC": '\033[m'}


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

    def __init__(self, logger_level: 'LoggingLevel' = "WARNING"):
        """
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # load configuration from logging.conf
        logging.config.fileConfig(Path(os.path.dirname(__file__)) / "logger.conf", defaults={'log_level': logger_level},
                                  disable_existing_loggers=False)

        # get logger
        self.logger = logging.getLogger('vdata.vlogger')

    @property
    def level(self) -> str:
        """
        Get the logging level.
        :return: the logging level.
        """
        return {10: 'DEBUG',
                20: 'INFO',
                30: 'WARNING',
                40: 'ERROR',
                50: 'CRITICAL'}[self.logger.level]

    @level.setter
    def level(self, logger_level: 'LoggingLevel') -> None:
        """
        Re-init the logger, for setting new minimal logging level
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if logger_level not in LoggingLevels:
            raise errors.VTypeError(f"Incorrect logging level '{logger_level}', should be in {LoggingLevels}")

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

        # Get the name of the file that called the logger for displaying where the message came from
        # if Tb.trace is None:
        #     frames = inspect.stack(0)
        #
        #     caller_filename = frames[0].filename
        #     index = 0
        #
        #     while index < len(frames) - 1 and (caller_filename.endswith("logger.py")
        #                                        or caller_filename.endswith("errors.py")):
        #         index += 1
        #         caller_filename = frames[index].filename
        #
        #     caller = os.path.splitext(os.path.basename(caller_filename))[0]
        #
        #     # return base message
        #     return f"[{caller}.py] {msg}"
        #
        # else:
        #     traceback.print_tb(Tb.trace)
        #     caller = ""
        #
        #     while Tb.trace is not None:
        #         caller = Tb.trace.tb_frame.f_code.co_filename
        #         Tb.trace = Tb.trace.tb_next
        #
        #     return f"[{os.path.basename(caller)} : {Tb.exception.__name__}] {msg}"

        return msg

    def debug(self, msg: str) -> None:
        """
        Log a debug message (level 10)

        :param msg: the message to be logged
        """
        self.logger.debug(colors["BGREY"] + self._getBaseMsg(msg) + colors["ENDC"])

    def info(self, msg: str) -> None:
        """
        Log an info message (level 20)

        :param msg: the message to be logged
        """
        self.logger.info(colors["TCYAN"] + self._getBaseMsg(msg) + colors["ENDC"])

    def warning(self, msg: str) -> None:
        """
        Log a warning message (level 30)

        :param msg: the message to be logged
        """
        self.logger.warning(colors["TORANGE"] + self._getBaseMsg(msg) + colors["ENDC"])

    def error(self, msg: str) -> None:
        """
        Log an error message (level 40)

        :param msg: the message to be logged
        """
        self.logger.error(colors["TRED"] + self._getBaseMsg(msg) + colors["ENDC"])
        quit()

    def uncaught_error(self, msg: str) -> None:
        """
        Log and uncaught (not originating from a custom error class) error message (level 40)

        :param msg: the message to be logged
        """
        traceback.print_tb(Tb.trace)

        last = None
        while Tb.trace is not None:
            last = Tb.trace.tb_frame
            Tb.trace = Tb.trace.tb_next

        # last.f_globals['__package__']
        self.logger.error(colors["TRED"] + f"[{last.f_globals['__name__'] if last is not None else 'UNCAUGHT'} :"
                                           f" {Tb.exception.__name__}] {msg}" + colors["ENDC"])

    def critical(self, msg: str) -> None:
        """
        Log a critical message (level 50)

        :param msg: the message to be logged
        """
        self.logger.critical(colors["TRED"] + colors["BBLACK"] + self._getBaseMsg(msg) + colors["ENDC"])


generalLogger = _VLogger()


def setLoggingLevel(log_level: 'LoggingLevel') -> None:
    """
    Set the logging level for package vdata.
    :param log_level: a logging level to set, in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    """
    generalLogger.level = log_level


def getLoggingLevel() -> str:
    """
    Get the logging level for package vdata.
    :return: the logging level for package vdata.
    """
    return generalLogger.level


# disable traceback messages, except if the logging level is set to DEBUG
def exception_handler(exception_type, exception, traceback_, debug_hook=sys.excepthook):
    Tb.trace = traceback_
    Tb.exception = exception_type

    if generalLogger.level == 'DEBUG':
        if not issubclass(exception_type, errors.VBaseError):
            generalLogger.uncaught_error(exception)
        debug_hook(exception_type, exception, traceback_)
    else:
        if not issubclass(exception_type, errors.VBaseError):
            generalLogger.uncaught_error(exception)
        else:
            print(exception)

    traceback.print_tb(traceback_)


sys.excepthook = exception_handler
