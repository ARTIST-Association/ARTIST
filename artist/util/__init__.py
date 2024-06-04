"""Bundles all classes that implement util functionality in ARTIST."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import colorlog


def set_logger_config(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_to_stdout: bool = True,
    colors: bool = True,
) -> None:
    """
    Set up the logger. Should only need to be done once. Generally, logging should only be done on the master rank.

    Parameters
    ----------
    level : int
        The default level for logging. Default is ``logging.INFO``.
    log_file : str | Path, optional
        The file to save the log to.
    log_to_stdout : bool
        A flag indicating if the log should be printed on stdout. Default is True.
    log_rank : bool
        A flag for prepending the MPI rank to the logging message. Default is False.
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
