from typing import TYPE_CHECKING

from query_common.parse.impl.lark import LarkParser
from query_common.values.types import ParserType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from query_common.parse.base import Parser

PARSER_CLASS_MAPPING: "Mapping[ParserType, type[Parser]]" = {
    ParserType.LARK: LarkParser,
}
