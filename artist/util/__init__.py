"""Bundle all classes that implement util functionality in ``ARTIST``."""

import logging
import sys
import time
from functools import wraps
from pathlib import Path

import colorlog


def set_logger_config(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    log_to_stdout: bool = True,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once. Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level : int
        The default level for logging. Default is ``logging.INFO``.
    log_file : str | Path | None
        The file to save the log to.
    log_to_stdout : bool
        A flag indicating if the log should be printed on stdout. Default is True.
    colors : bool
        A flag for using colored logs. Default is True.
    """
    # Get base logger for ARTIST.
    base_logger = logging.getLogger("artist")
    simple_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    if colors:
        formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(formatter)
    else:
        std_handler = logging.StreamHandler(stream=sys.stdout)
        std_handler.setFormatter(simple_formatter)

    if log_to_stdout:
        base_logger.addHandler(std_handler)
    if log_file is not None:
        log_file = Path(log_file)
        log_dir = log_file.parents[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(simple_formatter)
        base_logger.addHandler(file_handler)
    base_logger.setLevel(level)


def set_runtime_logger(
    log_file: str | Path = "runtime_log.txt",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a shared runtime logger that logs execution times of functions.

    Parameters
    ----------
    log_file : str | Path
        The file path to write runtime logs.
    level : int
        The logging level (default is logging.INFO).

    Returns
    -------
    logging.Logger
        The configured runtime logger.
    """
    logger_name = "artist.runtime"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def track_runtime(logger: logging.Logger):
    """
    Track and log start, finish, and duration of function execution.

    Parameters
    ----------
    logger : logging.Logger
        The runtime logger.

    Returns
    -------
    Callable
        The decorated function with runtime tracking.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"{func_name} started")
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(f"{func_name} finished in {duration:.3f}s")
            return result

        return wrapper

    return decorator


runtime_log = set_runtime_logger("./runtime_log.txt")
