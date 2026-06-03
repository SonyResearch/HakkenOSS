from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from query_common.entities.clauses.query import Query

ParserToken = "parser"


class Parser(ABC):
    @abstractmethod
    def parse_query(self, text: str) -> "Query":
        pass
