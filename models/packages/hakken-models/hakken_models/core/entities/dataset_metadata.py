from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    num_entities: int = Field(..., description="Total number of distinct entities in the dataset")
    num_relations: int = Field(..., description="Total number of distinct relations in the dataset")
    num_domains: int = Field(..., description="Total number of distinct domains in the dataset")
    num_timestamps: int = Field(
        ..., description="Total number of distinct timestamps in the dataset"
    )
