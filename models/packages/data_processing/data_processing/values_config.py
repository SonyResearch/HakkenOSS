from pydantic import BaseModel


class BaseFileConfig(BaseModel):
    path: str
    sep: str
    column_names: list[str]
    header: int | bool | None = None
    encoding: str = "utf-8"


class DataFiles(BaseModel):
    relations: list[BaseFileConfig]
    ontologies: list[BaseFileConfig] | None = None
