from pydantic import BaseModel, Field


class Relation(BaseModel):
    name: str = Field(description="The name of the relation")

    def to_string(self) -> str:
        return self.name

    def metadata(self) -> dict:
        return {"name": self.name}
