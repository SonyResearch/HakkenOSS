class InvalidElementsError(Exception):
    def __init__(self, count: int, expected_count: int):
        super().__init__(f"Expected {expected_count} elements, got {count}")


class MissingReferenceKgError(ValueError):
    """Raised when the reference knowledge graph is not provided."""

    def __init__(self, message="Reference kg not provided."):
        self.message = message
        super().__init__(self.message)


class InvalidColumnError(KeyError):
    """Raised when a non-existent column is chosen."""

    def __init__(self, message="Column not present in dataframe"):
        self.message = message
        super().__init__(self.message)
