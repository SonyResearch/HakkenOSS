from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from sys import stdout
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from logfmter import Logfmter

if TYPE_CHECKING:
    from hakken_utils.core.contract.logger import PossibleLabelValues

ExcInfo = tuple[type[BaseException], BaseException, TracebackType]

ANSI_PER_LEVEL = {
    "debug": "\u001b[94m",  # blue
    "info": "\u001b[32m",  # green
    "warning": "\u001b[33m",  # yellow
    "error": "\u001b[31m",  # red
    "critical": "\u001b[35m",  # magenta
}
IS_TTY = stdout.isatty()


def comma_separated_env_to_list(key: str) -> list[str]:
    """
    Convert a comma-separated environment variable to a list of strings.

    Args:
        key (str): The environment variable key.

    Returns:
        List[str]: A list of strings obtained from the environment variable.
    """
    value = os.getenv(key)
    if value is None:
        return []
    return value.split(",")


WHITELISTED_LABELS = set(comma_separated_env_to_list("SPAICE_LOG_LOGFMT_WHITELISTED_LABELS"))
BLACKLISTED_LABELS = set(comma_separated_env_to_list("SPAICE_LOG_LOGFMT_BLACKLISTED_LABELS"))
LABEL_KEYS_ENABLED = os.getenv("SPAICE_LOG_LOGFMT_LABEL_KEYS_ENABLED", "1") in ["1", "True", "true"]


class LogFmtFormatter(Logfmter):
    """
    A log formatter that formats logs in logfmt format with support for common
    labels and color-coded log levels.
    """

    common_labels: ContextVar[dict[str, Any] | None] = ContextVar(
        "LogFmtFormatter.labels", default=None
    )
    possible_label_values: ContextVar[dict[str, PossibleLabelValues] | None] = ContextVar(
        "LogFmtFormatter.possible_label_values", default=None
    )

    def __init__(
        self,
        keys: list[str] | None = None,
        mapping: dict[str, str] | None = None,
        datefmt: str | None = None,
        common_labels: dict[str, Any] | None = None,
        possible_label_values: dict[str, PossibleLabelValues] | None = None,
    ):
        """
        Initialize the LogFmtFormatter with optional keys, mapping, date format, common labels, and
        possible label values.

        Args:
         keys (List[str]): List of keys to include in the log output.
         mapping (Dict[str, str]): Mapping of custom key names to log record attributes.
         datefmt (str, optional): Date format string.
         common_labels (Dict[str, Any], optional): Common labels to include in each log record.
         possible_label_values (Dict[str, PossibleLabelValues], optional): Possible values for lbl.
        """
        resolved_keys = keys or ["at"]
        resolved_mapping = mapping or {"at": "levelname"}
        resolved_common_labels = common_labels or {}
        resolved_possible_label_values = possible_label_values or {}

        self.common_labels.set(resolved_common_labels)
        self.possible_label_values.set(
            {
                **self.__process_possible_label_values(resolved_possible_label_values),
                **{"level": {"max_length": 8}},
            }
        )
        if datefmt is not None:
            super().__init__(resolved_keys, resolved_mapping, datefmt)
        else:
            super().__init__(resolved_keys, resolved_mapping)

    def get_extra(self, record: logging.LogRecord) -> dict:  # type: ignore
        """
        Inject common labels into the log record.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            dict: The updated log record dictionary with common labels.
        """
        # inject our own common labels
        record.__dict__ = {
            **{
                key: value
                for key, value in (self.common_labels.get() or {}).items()
                if key not in record.__dict__
            },
            **record.__dict__,
        }
        # and let the parent handle the rest
        return super().get_extra(record)

    def __process_possible_label_values(
        self,
        possible_label_values: dict[str, PossibleLabelValues],
    ) -> dict[str, PossibleLabelValues]:
        """
        Process possible label values to determine the maximum length of each label.

        Args:
            possible_label_values (Dict[str, PossibleLabelValues]): Dictionary of possible label
            values.

        Returns:
            Dict[str, PossibleLabelValues]: Dictionary with processed maximum length for each label.
        """

        result: dict[str, PossibleLabelValues] = {}

        for label, value in possible_label_values.items():
            if "max_length" in value:
                result[label] = {"max_length": value["max_length"]}
            elif "values" in value and len(value["values"]) > 0:
                result[label] = {"max_length": len(max(value["values"], key=len))}

        return result

    @classmethod
    def format_string(cls, value: str) -> str:
        """
        Format a string value for log output, escaping double quotes and quoting if necessary.

        Args:
            value (str): The string value to format.

        Returns:
            str: The formatted string.
        """
        """We override this method to allow newlines to be printed in
        the log message without escaping them"""
        needs_dquote_escaping = '"' in value
        needs_quoting = " " in value or "=" in value or "\n" in value

        if needs_dquote_escaping:
            value = value.replace('"', '\\"')

        if needs_quoting:
            value = f'"{value}"'

        return value if value else '""'

    @classmethod
    def format_value(cls, value):
        """
        Format a value for log output, applying color coding for log levels.

        Args:
            value: The value to format.

        Returns:
            str: The formatted value.
        """
        if value in logging._nameToLevel:
            # for consistency between different programming languages we show the level in lowercase
            level_lower = value.lower()
            if IS_TTY:
                # let's color it for human consumption
                return ANSI_PER_LEVEL[level_lower] + level_lower + "\033[0m"
            return level_lower

        return super().format_value(value)

    @classmethod
    def __get_formatted_key_value(
        cls, key: Any, value: Any, possible_label_values: dict[str, PossibleLabelValues]
    ) -> str:
        """
        Get the formatted key-value pair for log output, considering possible label values.

        Args:
            key (Any): The key.
            value (Any): The value.
            possible_label_values (Dict[str, PossibleLabelValues]): Dictionary of possible label
                values.

        Returns:
            str: The formatted key-value pair.
        """
        amount_of_spaces = 1
        if key in possible_label_values:
            amount_of_spaces += possible_label_values[key]["max_length"] - len(value)

        result = ""

        if LABEL_KEYS_ENABLED:
            result += f"{key}={cls.format_value(value)}"
        else:
            result += f"{cls.format_value(value)}"

        result += amount_of_spaces * " "

        return result

    @classmethod
    def __key_is_blacklisted(cls, key: Any) -> bool:
        if len(WHITELISTED_LABELS) > 0 and key not in WHITELISTED_LABELS:
            return True

        return key in BLACKLISTED_LABELS

    @classmethod
    def format_params(  # type: ignore
        cls, params: dict, possible_label_values: dict[str, PossibleLabelValues]
    ) -> str:
        """
        Format log parameters, ensuring 'msg' appears last for visibility.

        Args:
            params (dict): The log parameters.
            possible_label_values (Dict[str, PossibleLabelValues]): Dictionary of possible label
                values.

        Returns:
            str: The formatted parameters.
        """
        if "msg" in params:
            # forces msg to always appear last for visibility reasons
            msg = params["msg"]
            del params["msg"]
            params["msg"] = msg

        result = ""
        for key, value in params.items():
            if not cls.__key_is_blacklisted(key):
                result += cls.__get_formatted_key_value(key, value, possible_label_values)

        return result

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a string in logfmt format.

        Args:
            record (logging.LogRecord): The log record.

        Returns:
            str: The formatted log record as a string.
        """
        # most of this was copied over from the parent class
        # because we need to do our own things in-between

        if "asctime" in self.keys or "asctime" in self.mapping.values():
            record.asctime = self.formatTime(record, self.datefmt)

        if isinstance(record.msg, dict):
            params = {self.normalize_key(key): value for key, value in record.msg.items()}
        else:
            extra = self.get_extra(record)
            params = {"msg": record.getMessage(), **extra}

        result_str = ""

        for key in self.keys:
            attribute = key

            if key in self.mapping:
                attribute = self.mapping[key]

            if not hasattr(record, attribute) or LogFmtFormatter.__key_is_blacklisted(key):
                continue

            result_str += LogFmtFormatter.__get_formatted_key_value(
                key, getattr(record, attribute), self.possible_label_values.get() or {}
            )

        formatted_params = self.format_params(params, self.possible_label_values.get() or {})
        if formatted_params:
            result_str += formatted_params

        if record.exc_info:
            exc_info = cast("ExcInfo", record.exc_info)

            result_str += f"exc_info={self.format_exc_info(exc_info)} "

        if result_str.endswith(" "):
            result_str = result_str[:-1]

        return result_str
