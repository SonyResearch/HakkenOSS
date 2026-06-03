from typing import TYPE_CHECKING

from dependency_injector.wiring import Provide, inject

if TYPE_CHECKING:
    from query_common.entities.clauses.query import Query
    from query_common.parse.base import Parser


@inject
def parse_query(query_string: str, parser: "Parser" = Provide["parser"]) -> "Query":
    return parser.parse_query(query_string)
