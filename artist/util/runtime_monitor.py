import logging
import time
from functools import wraps
from pathlib import Path

class RuntimeLogger:
    """Separate runtime logger that writes function execution info to a file."""
    def __init__(self, log_file: str | Path = "runtime_log.txt"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._created_loggers = {}

    def get_logger(self, name: str):
        """Returns a logger with module name, configured for runtime log file."""
        if name in self._created_loggers:
            return self._created_loggers[name]

        logger = logging.getLogger(name + "_runtime")  # avoid collision with other loggers
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(file_handler)

        self._created_loggers[name] = logger
        return logger

    def track_runtime(self, logger: logging.Logger):
        """Decorator to log function start, finish, and duration."""
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
