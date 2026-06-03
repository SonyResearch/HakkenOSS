import logging
import os
from contextvars import ContextVar
from typing import Any, ClassVar, cast

from hakken_utils.core.contract.logger import ILogger, PossibleLabelValues
from hakken_utils.impl.logging.log_fmt_formatter import LogFmtFormatter
from hakken_utils.impl.logging.raw_formatter import RawFormatter


def set_formatter(
    logger: logging.Logger,
    labels: dict[str, Any] | None = None,
    possible_label_values: dict[str, PossibleLabelValues] | None = None,
):
    labels = labels or {}
    possible_label_values = possible_label_values or {}
    env_log_format_raw = os.getenv("HAKKEN_LOG_FORMAT")
    env_log_format = env_log_format_raw.lower() if env_log_format_raw is not None else "logfmt"
    log_format = env_log_format if env_log_format in ["plain", "logfmt"] else "logfmt"

    formatter = (
        RawFormatter()
        if log_format == "plain"
        else LogFmtFormatter(
            keys=["ts", "level"],
            mapping={
                # mappings for consistency between programming languages:
                "ts": "asctime",  # renames asctime to ts
                "level": "levelname",
            },
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            common_labels=labels,
            possible_label_values=possible_label_values,
        )
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    if len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    logger.addHandler(handler)


def get_log_level() -> str:
    env_log_level = (
        os.getenv("HAKKEN_LOG_LEVEL", "").upper()
        if os.getenv("HAKKEN_LOG_LEVEL") is not None
        else None
    )

    return env_log_level if env_log_level in logging._nameToLevel else "INFO"


class HakkenLogger(ILogger):
    labels: ContextVar[dict[str, Any] | None] = ContextVar("HakkenLogger.labels", default=None)
    possible_label_values: ContextVar[dict[str, PossibleLabelValues] | None] = ContextVar(
        "HakkenLogger.possible_label_values", default=None
    )

    def __init__(self, name):
        super().__init__(name)

    def __set_formatter(self):
        set_formatter(self, self.labels.get(), self.possible_label_values.get())

    def initialize(
        self,
        labels: dict[str, Any] | None = None,
        possible_label_values: dict[str, PossibleLabelValues] | None = None,
    ):
        self.propagate = False  # disable duplicate log entries from the root level
        self.setLevel(get_log_level())
        self.__add_labels(labels)
        self.__set_possible_label_values(possible_label_values)
        self.__set_formatter()

    def __set_possible_label_values(
        self, possible_label_values: dict[str, PossibleLabelValues] | None = None
    ):
        self.possible_label_values.set(possible_label_values or {})

    def __add_labels(self, labels: dict[str, Any] | None = None):
        self.labels.set({**(self.labels.get() or {}), **(labels or {})})

    def add_labels(self, labels: dict[str, Any] | None = None):
        self.__add_labels(labels)
        self.__set_formatter()  # formatter needs to be updated with the new labels

    def remove_labels(self, labels: list[str]):
        current_labels = dict(self.labels.get() or {})
        for label in labels:
            current_labels.pop(label, None)
        self.labels.set(current_labels)

        self.__set_formatter()

    def set_labels(self, labels: dict[str, Any] | None = None):
        self.labels.set(labels or {})
        self.__set_formatter()

    def extend(self, name: str, labels: dict[str, Any] | None = None) -> "HakkenLogger":
        extended_logger = HakkenLogger(name)
        extended_logger.initialize(
            labels={**(self.labels.get() or {}), **(labels or {})},
            possible_label_values=self.possible_label_values.get() or {},
        )
        Logging.logger_by_name[name] = extended_logger
        return extended_logger


class Logging:
    logger_by_name: ClassVar[dict[str, HakkenLogger]] = {}
    root_labels: ClassVar[dict[str, Any]] = {}

    @classmethod
    def getLogger(  # noqa: N802
        cls,
        name: str,
        labels: dict[str, Any] | None = None,
        possible_label_values: dict[str, PossibleLabelValues] | None = None,
    ) -> HakkenLogger:
        labels = labels or {}
        possible_label_values = possible_label_values or {}
        if name is None or name == "root":
            raise Exception("Please use init_root_logger to initialize the root logger")

        if name in cls.logger_by_name:
            return cls.logger_by_name[name]

        # ref: https://stackoverflow.com/a/50401350
        logging_class = logging.getLoggerClass()
        # thread safety
        logging._acquireLock()  # type: ignore
        try:
            logging.setLoggerClass(HakkenLogger)
            logger = cast("HakkenLogger", logging.getLogger(name))
            logger.initialize(labels=labels, possible_label_values=possible_label_values)
            # switch back the logging class
            logging.setLoggerClass(logging_class)
            cls.logger_by_name[name] = logger
            return logger
        finally:
            logging._releaseLock()  # type: ignore

    @classmethod
    def init_root_logger(cls, labels: dict[str, Any] | None = None) -> logging.Logger:
        labels = labels or {}
        # root logger's class cannot be customized
        logging.basicConfig(level=get_log_level())
        root_logger = logging.getLogger()
        set_formatter(root_logger, labels)
        cls.root_labels = labels
        return root_logger

    @classmethod
    def remove_root_labels(cls, labels: list[str]):
        for label in labels:
            cls.root_labels.pop(label, None)

        set_formatter(logging.getLogger(), cls.root_labels)

    @classmethod
    def add_root_labels(cls, labels: dict[str, Any] | None = None):
        cls.root_labels = {**cls.root_labels, **(labels or {})}

        set_formatter(logging.getLogger(), cls.root_labels)

    @classmethod
    def set_root_labels(cls, labels: dict[str, Any] | None = None):
        cls.root_labels = labels or {}

        set_formatter(logging.getLogger(), cls.root_labels)
