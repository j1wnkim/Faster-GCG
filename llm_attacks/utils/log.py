import logging
import logging.handlers
import sys
from logging import Handler, Logger
from typing import Optional, TextIO


# --------------- Copied from timm/utils/log.py ---------------
class FormatterNoInfo(logging.Formatter):
    def __init__(
        self,
        fmt="[%(name)s::%(lineno)d:%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ):
        logging.Formatter.__init__(self, fmt, datefmt=datefmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(
    default_level=logging.INFO, log_path="", name: Optional[str] = None, no_stream=False
):
    """Setup default logging format.

    Args:
        default_level (logging._Level):
            Log level. Default: `logging.INFO`.
        log_path (str):
            Log path. If specified, also log to file. Default: ''.
        name (str | None):
            Logger name. If None, setup root logger.
        no_stream (bool):
            If set, it will not log to stream.

    Returns:
        Logger: The logger configured.
    """
    logger = logging.getLogger(name)
    if name is not None:
        if id(logger) == id(logging.root):
            logging.error("A name is specified but root logger is returned")
        # Logger has it own handlers, do not propagate
        logger.propagate = False
    logger.setLevel(default_level)
    if not no_stream:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(FormatterNoInfo())
        # Handler has it own level,
        # log should pass both the logger's level and the handler's level
        logger.addHandler(console_handler)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(8 << 20), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s [%(name)s:%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger


# --------------- Custom ---------------
def setup_logger(
    name: Optional[str] = None,
    filename: Optional[str] = None,
    stream: Optional[TextIO] = sys.stderr,
    timestamp=False,
    **kwargs,
):
    """To setup as many loggers as you want.

    Args:
        name: logger name
        filename: log file name, set None to disable
        stream: log to stream, set None to disable
        timestamp: add timestamp to log output
        **kwargs: any other keyword arguments to be passed into `logging.basicConfig()`
    """
    if timestamp:
        # Default format and date format with timestamp
        kwargs.setdefault(
            "format", "[%(asctime)s,%(name)s,%(levelname)s] - %(message)s"
        )
        kwargs.setdefault("datefmt", "%Y/%m/%d %H:%M:%S")
    else:
        kwargs.setdefault("format", "%(message)s")
    default_handlers = []
    if stream is not None:
        console_handler = logging.StreamHandler(stream)
        default_handlers.append(console_handler)
    if filename is not None:
        # file_handler = logging.FileHandler(filename)
        file_handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=(8 << 20), backupCount=3
        )
        default_handlers.append(file_handler)
    kwargs.setdefault("handlers", default_handlers)
    assert len(kwargs["handlers"]), (
        "Some handlers should be specified by 'stream', 'filename' or 'handlers'"
    )
    root_logger = logging.root
    logger = logging.getLogger(name)
    if name is not None:
        assert id(logger) != id(logging.root), (
            "A name is specified but root logger is returned"
        )
    logging.root = logger
    logging.basicConfig(**kwargs)  # This only applies to root
    logging.root = root_logger
    # Important! Otherwise the log will propagate to parent logger's handlers
    logger.propagate = False
    return logger


def suppress_logger(logger: Logger, handlers: list[Handler] | None = None):
    logger.propagate = False
    logger.handlers = handlers or []


def log_contents(logger: Optional[Logger], lines: list, level=logging.INFO):
    """Use logger if not None, else use root logger."""
    for line in lines:
        if logger is not None:
            logger.log(level, line)
        else:
            logging.log(level, line)
