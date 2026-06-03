from typing import Any

from data_processing.values import LIST_TO_STRING_SEPARATOR


def aggressive_quote_clean(text: str) -> str:
    # Remove ALL quotes and clean up any remaining escape sequences
    return text.replace('"', "").replace("\\", "")


def convert_list_to_string(x: list[Any], separator: str = LIST_TO_STRING_SEPARATOR):
    return separator.join(map(str, x))


def chunk_list(x: list[Any], chunk_size=200):
    for i in range(0, len(x), chunk_size):
        yield x[i : i + chunk_size]
