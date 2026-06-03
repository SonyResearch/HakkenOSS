from enum import StrEnum


class TKGEngine(StrEnum):
    IN_MEMORY = "in_memory"

    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]
