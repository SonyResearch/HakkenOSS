from typing import Any


class InvalidConfigTypeError(TypeError):
    def __init__(self, expected_type: str, expected_class: type, received_config: Any) -> None:
        self.expected_type = expected_type
        self.expected_class = expected_class
        self.received_class = type(received_config)

        message = (
            f"{expected_type} requires configuration of type {expected_class.__name__}, "
            f"got {self.received_class.__name__} instead"
        )
        super().__init__(message)
