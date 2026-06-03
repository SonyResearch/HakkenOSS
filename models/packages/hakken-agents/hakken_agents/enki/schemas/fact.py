from pydantic import BaseModel, Field

from hakken_agents.enki.schemas.relation import Relation


class Quantity(BaseModel):
    value: float = Field(description="The value of the quantity")
    unit: str = Field(description="The unit of the quantity")


class Fact(BaseModel):
    subject_name: str = Field(description="The name of the subject")
    subject_domain: str = Field(description="The domain of the subject")
    subject_quantity: Quantity | None = Field(description="The quantity of the subject")

    relation: Relation = Field(description="The relation between the subject and object")
    object_name: str = Field(description="The name of the object")
    object_domain: str = Field(description="The domain of the object")
    object_quantity: Quantity | None = Field(description="The quantity of the object")
