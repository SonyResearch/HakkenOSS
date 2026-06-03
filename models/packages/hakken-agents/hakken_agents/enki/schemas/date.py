from pydantic import BaseModel, Field


class Date(BaseModel):
    year: int = Field(description="The year of the date")
    month: int | None = Field(description="The month of the date")
    day: int | None = Field(description="The day of the date")
