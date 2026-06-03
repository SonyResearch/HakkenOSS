from typing import Literal

from typing_extensions import TypedDict


class PromptMessage(TypedDict):
    role: Literal["system", "user"]
    content: str
