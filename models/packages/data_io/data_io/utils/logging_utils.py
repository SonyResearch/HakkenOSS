import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from data_io.utils.path_utils import repo_folder

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LogHandler(Enum):
    """
    Types of log handlers.
    """

    FILE = "FILE"
    CONSOLE = "CONSOLE"
    ALL = "ALL"


class LogFormat(Enum):
    """
    Common logging formats.
    """

    BASIC = "[%(asctime)s %(levelname)s] %(message)s"
    EXTENDED = "%(asctime)s %(levelname)-7s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    TEST = "%(message)s"


def setup_logger(
    handler: LogHandler = LogHandler.CONSOLE,
    level: int | str = "INFO",
    format: LogFormat | str = LogFormat.BASIC,
):
    """
    Setup a logger, on file, console or both
    """
    handlers: list[logging.Handler] = []
    if handler in [LogHandler.FILE, LogHandler.ALL]:
        log_path = Path(f"{repo_folder()}/.logs")
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(
                filename=str(log_path / f"log_{datetime.now():%Y-%m-%d_%H:%M:%S}.txt"),
                mode="w",
                encoding="utf-8",
            )
        )
    if handler in [LogHandler.CONSOLE, LogHandler.ALL]:
        handlers.append(logging.StreamHandler())

    if isinstance(format, LogFormat):
        format = format.value

    logging.basicConfig(format=format, level=level, handlers=handlers)
