import re

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator


class Domain(BaseModel):
    name: str = Field(description="The name or identifier of the domain")

    @staticmethod
    def clean_level(level: str) -> str:
        """Normalize domain name: lowercase levels, letters and underscore only."""
        if not isinstance(level, str):
            return level

        return re.sub(r"[^a-z_]", "_", level.strip().lower())

    @staticmethod
    def clean_name(name: str) -> str:
        """Normalize domain name: lowercase levels, letters and underscore only."""
        if not isinstance(name, str):
            return name
        return "/".join([Domain.clean_level(segment) for segment in name.split("/")])

    @field_validator("name", mode="before")
    @classmethod
    def name_cleaned(cls, v: str) -> str:
        if isinstance(v, str):
            return cls.clean_name(v)
        return v

    def to_string(self) -> str:
        return f"{self.name}"

    def update_level(self, level: int, value: str) -> None:
        levels = self.get_levels()
        levels[level - 1] = Domain.clean_level(value)
        self.name = "/".join(levels)

    def num_levels(self) -> int:
        return len(self.get_levels())

    def get_levels(self) -> list[str]:
        return self.name.split("/")

    def get_level(self, level: int) -> str:
        if level < 1:
            raise ValueError("Level must be greater than 0")
        levels = self.get_levels()
        return levels[level - 1]

    def metadata(self) -> dict:
        levels = self.get_levels()
        metadata = {}
        for i, level in enumerate(levels):
            metadata[f"level_{i + 1}"] = level

        return metadata

    def to_document(self) -> Document:
        return Document(
            page_content=self.to_string(),
            metadata=self.metadata(),
        )
