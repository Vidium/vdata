from __future__ import annotations

import logging.config
import os
import sys
import traceback
import warnings
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Optional, Type

from vdata.IO import errors

warnings.simplefilter(action="ignore", category=FutureWarning)


class LoggingLevel(Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Color:
    TCYAN = "\033[36m"
    TORANGE = "\033[33m"
    TRED = "\033[31m"
    BBLACK = "\033[40m"
    BGREY = "\033[100m"
    ENDC = "\033[0m"


class Tb:
    trace: Optional[TracebackType] = None
    exception: Type[BaseException] = BaseException


def callable_msg(level: LoggingLevel) -> Callable[[Callable[..., None]], Callable[..., Any]]:
    def inner(func: Callable[[_VLogger, str | Callable[[], str]], None]) -> Callable[..., Any]:
        def wrapper(self: _VLogger, msg: str | Callable[[], str]) -> None:
            if not self.logger.isEnabledFor(level.value):
                return None

            if callable(msg):
                msg = msg()

            return func(self, msg)

        return wrapper

    return inner


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

    def __init__(self, logger_level: LoggingLevel = LoggingLevel.WARNING):
        """
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # load configuration from logging.conf
        logging.config.fileConfig(
            Path(os.path.dirname(__file__)) / "logger.conf",
            defaults={"log_level": logger_level.name},
            disable_existing_loggers=False,
        )

        # get logger
        self.logger = logging.getLogger("vdata.vlogger")

    @property
    def level(self) -> LoggingLevel:
        """
        Get the logging level.
        :return: the logging level.
        """
        return LoggingLevel(self.logger.level)

    @level.setter
    def level(self, log_level: LoggingLevel | str) -> None:
        """
        Re-init the logger, for setting new minimal logging level
        :param logger_level: minimal log level for the logger. (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = _as_log_level(log_level)

        self.logger.setLevel(log_level.name)
        for handler in self.logger.handlers:
            handler.setLevel(log_level.name)

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

    @callable_msg(LoggingLevel.DEBUG)
    def debug(self, msg: str) -> None:
        """
        Log a debug message (level 10)

        :param msg: the message to be logged
        """
        self.logger.debug(Color.BGREY + self._getBaseMsg(msg) + Color.ENDC)

    @callable_msg(LoggingLevel.INFO)
    def info(self, msg: str) -> None:
        """
        Log an info message (level 20)

        :param msg: the message to be logged
        """
        self.logger.info(Color.TCYAN + self._getBaseMsg(msg) + Color.ENDC)

    @callable_msg(LoggingLevel.WARNING)
    def warning(self, msg: str) -> None:
        """
        Log a warning message (level 30)

        :param msg: the message to be logged
        """
        self.logger.warning(Color.TORANGE + self._getBaseMsg(msg) + Color.ENDC)

    @callable_msg(LoggingLevel.ERROR)
    def error(self, msg: str) -> None:
        """
        Log an error message (level 40)

        :param msg: the message to be logged
        """
        self.logger.error(Color.TRED + self._getBaseMsg(msg) + Color.ENDC)
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

        self.logger.error(
            f"[{last.f_globals['__name__'] if last is not None else 'UNCAUGHT'} :" f" {Tb.exception.__name__}] {msg}"
        )

    @callable_msg(LoggingLevel.CRITICAL)
    def critical(self, msg: str) -> None:
        """
        Log a critical message (level 50)

        :param msg: the message to be logged
        """
        self.logger.critical(Color.TRED + Color.BBLACK + self._getBaseMsg(msg) + Color.ENDC)


generalLogger = _VLogger()


def _as_log_level(log_level: LoggingLevel | str) -> LoggingLevel:
    if not isinstance(log_level, LoggingLevel):
        try:
            return LoggingLevel[log_level]

        except KeyError as e:
            raise KeyError(
                f"Incorrect logging level '{log_level}', " f"should be in {[ll.value for ll in LoggingLevel]}"
            ) from e

    return log_level


def setLoggingLevel(log_level: LoggingLevel | str) -> None:
    """
    Set the logging level for package vdata.
    :param log_level: a logging level to set, in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    """
    generalLogger.level = _as_log_level(log_level)


def getLoggingLevel() -> LoggingLevel:
    """
    Get the logging level for package vdata.
    :return: the logging level for package vdata.
    """
    return generalLogger.level


# disable traceback messages, except if the logging level is set to DEBUG
def exception_handler(
    exception_type: Type[BaseException],
    exception: BaseException,
    traceback_: TracebackType,
    debug_hook: Callable[[type[BaseException], BaseException, TracebackType | None], Any] = sys.excepthook,
) -> None:
    Tb.trace = traceback_
    Tb.exception = exception_type

    if generalLogger.level == LoggingLevel.DEBUG:
        if not issubclass(exception_type, errors.VBaseError):
            generalLogger.uncaught_error(str(exception))
        debug_hook(exception_type, exception, traceback_)
    else:
        if not issubclass(exception_type, errors.VBaseError):
            generalLogger.uncaught_error(str(exception))
        else:
            print(exception)

    traceback.print_tb(traceback_)


sys.excepthook = exception_handler  # type: ignore[assignment]
