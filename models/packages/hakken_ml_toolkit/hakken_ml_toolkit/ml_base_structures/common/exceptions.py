from pathlib import Path


class TripleNotFoundError(Exception):
    """Raised when the tensor contains non-finite values."""

    pass


class InvalidDTypeError(Exception):
    """Raised when the tensor has an invalid data type."""

    def __init__(self, actual_value: str, expected_value: str):
        super().__init__(f"Expected {expected_value} dype, got {actual_value}")


class InvalidDimensionError(Exception):
    """Raised when the tensor has invalid dimensions."""

    def __init__(self, count: int, expected: int | str):
        super().__init__(f"Expected {expected} dimensions, got {count}")


class InvalidColumnError(Exception):
    """Raised when the tensor contains invalid columns."""

    pass


class NegativeValueError(Exception):
    """Raised when the tensor contains negative values."""

    pass


class NonFiniteValueError(Exception):
    """Raised when the tensor contains non-finite values."""

    pass


class EmptyListError(Exception):
    """Raised when the tensor contains an empty list."""

    pass


class InvalidFormatError(Exception):
    """Raised when data format is invalid."""

    pass


class MappingNotFoundError(Exception):
    """Raised when mapping is invalid."""

    def __init__(self, folder: str | Path) -> None:
        super().__init__(f"Mapping not found at {folder}")


class SplitNotInTriplesError(Exception):
    """Raised when the split value is not present within triples dictionary."""

    def __init__(self, split: str, valid_splits: list[str] | None = None) -> None:
        self.split = split
        self.valid_splits = valid_splits or ["train", "test", "val", "all"]
        self.message = f"Split '{self.split}' not found in triples dictionary."
        if self.valid_splits:
            splits_str = "', '".join(self.valid_splits)
            self.message += f" Valid splits are: '{splits_str}'"

        super().__init__(self.message)


class SRObjectError(Exception):
    """Raised when the subject, relation and object are invalid."""

    pass


class InvalidIndexError(Exception):
    """Raised when index type is invalid."""

    pass


class InvalidElementsError(Exception):
    def __init__(self, count: int, expected_count: int):
        super().__init__(f"Expected {expected_count} elements, got {count}")


class EntityNotInTripleError(Exception):
    def __init__(self, entity: int | str):
        super().__init__(f"Entity {entity} not found in triple")


class InvalidTriplesDictKeyError(Exception):
    def __init__(self, invalid_key: str, valid_keys: list[str]) -> None:
        self.invalid_key: str = invalid_key
        self.valid_keys: list[str] = valid_keys
        message: str = (
            f"Invalid key '{invalid_key}' in triples_dict. Valid keys are: {', '.join(valid_keys)}"
        )
        super().__init__(message)
