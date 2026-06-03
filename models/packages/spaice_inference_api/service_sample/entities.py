from pydantic import BaseModel, StrictStr


class MyInferenceRequest(BaseModel):
    hello: StrictStr


class MyInferenceResponse(BaseModel):
    hello: str
