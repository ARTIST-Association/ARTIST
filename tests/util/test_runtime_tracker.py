import logging
import pathlib
import re
import time
from typing import cast

import pytest

from artist.util import set_runtime_logger, track_runtime


@pytest.fixture
def runtime_logger(tmp_path: pathlib.Path) -> logging.Logger:
    """
    Pytest fixture that returns a runtime logger writing to a temporary file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path to the logs.

    Returns
    -------
    logging.Logger
        A temporary logger.
    """
    log_file = tmp_path / "runtime_test.log"
    logger = set_runtime_logger(log_file=log_file, level=logging.INFO)
    return logger


def test_track_runtime_with_fixture(runtime_logger: logging.Logger) -> None:
    """
    Test the runtime tracker decorator.

    Parameters
    ----------
    runtime_logger : logging.Logger
        Temporary runtime logger.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """

    @track_runtime(runtime_logger)
    def _dummy_function(x, y):
        time.sleep(0.05)
        return x + y

    result = _dummy_function(2, 3)
    assert result == 5

    file_handler = cast(logging.FileHandler, runtime_logger.handlers[0]).baseFilename
    log_contents = pathlib.Path(file_handler).read_text()

    assert "dummy_function started" in log_contents
    assert "dummy_function finished in" in log_contents
    match = re.search(r"finished in (\d+\.\d+)s", log_contents)
    assert match is not None
