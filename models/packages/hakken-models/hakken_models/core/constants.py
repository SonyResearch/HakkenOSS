from strenum import StrEnum

DEFAULT_NEGATIVE_SCORE = -1000.0


class FactComponent(StrEnum):
    SUBJECT = "subject"
    RELATION = "relation"
    OBJECT = "object"


class ModelType(StrEnum):
    THIGER = "thiger"
    KGE = "kge"
    SEGAL = "segal"


class MissingPolicy(StrEnum):
    RAISE = "raise"
    ZERO = "zero"


class LLMProvider(StrEnum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    OLLAMA = "ollama"
