from pydantic import BaseModel


class RelationPrediction(BaseModel):
    logits: list
    probs: list | None
