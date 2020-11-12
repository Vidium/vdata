# coding: utf-8
# Created on 11/6/20 5:55 PM
# Author : matteo

# ====================================================
# imports
import os
# import sys
import logging.config
import inspect
# import traceback
from pathlib import Path

from ..NameUtils import LoggingLevel


# ====================================================
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

    def __init__(self, logger_level: LoggingLevel = "INFO"):
        """
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_level = logger_level

        # load configuration from logging.conf
        logging.config.fileConfig(Path(os.path.dirname(__file__)) / "logger.conf", defaults={'log_level': self.log_level})

        # get logger
        self.logger = logging.getLogger('root.vlogger')

    def set_level(self, logger_level: LoggingLevel) -> None:
        """
        Re-init the logger, for setting new minimal logging level
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger.setLevel(logger_level)

    @staticmethod
    def _getBaseMsg(msg: str) -> str:
        """
        Build the message to log with format <[fileName.py] msg>

        :param msg: the message to be logged
        :return: the formatted message
        """
        # Get the name of the file that called the logger for displaying where the message came from
        frames = inspect.stack(0)
        caller_filename = frames[0].filename
        index = 0

        while index < len(frames) - 1 and caller_filename.endswith("logger.py"):
            index += 1
            caller_filename = frames[index].filename

        # TODO : should go 1 up in the stack to get the file name from which the error was called, it stops at error.py here, why ???
        caller = os.path.splitext(os.path.basename(caller_filename))[0]

        # return base message
        return f"[{caller}.py] {msg}"

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

    def critical(self, msg: str) -> None:
        """
        Log a critical message (level 50)

        :param msg: the message to be logged
        """
        self.logger.critical(self._getBaseMsg(msg))


generalLogger = _VLogger()
