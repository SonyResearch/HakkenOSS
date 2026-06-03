from typing import Any

from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    size: int
    levels: dict[str, Any]  # E.g. {"level_1": "HECHOS", "level_2": "section"}
    page_idx: int
    doc_id: str

    @property
    def levels_str(self) -> str:
        return "\n".join(f"{value}" for _level, value in self.levels.items())


class ParsedDocument(BaseModel):
    chunks: list[Chunk]
    absolute_path: str
    doc_id: str
