from typing import TYPE_CHECKING

from query_common.values.class_mapping import PARSER_CLASS_MAPPING

if TYPE_CHECKING:
    from query_common.parse.base import Parser
    from query_common.values.types import ParserType


def create_parser(parser_type: "ParserType") -> "Parser":
    return PARSER_CLASS_MAPPING[parser_type]()
