class NoSamplesError(Exception):
    pass


class UnknownReductionError(Exception):
    """Raised when the reduction value set is unknown."""

    pass


class UnknownAverageError(Exception):
    """Raised when the reduction value set is unknown."""

    def __init__(self, value: str) -> None:
        super().__init__(f"Unknown average {value}")


class UnknownMetricError(Exception):
    """Raised when the reduction value set is unknown."""

    def __init__(self, value: str) -> None:
        super().__init__(f"Unknown metric {value}")


class TopKLargerThanNumberOfItemsError(Exception):
    """Raised when the specified top_k parameter is larger than the number of available items."""

    def __init__(self, top_k: int, number_of_items: int) -> None:
        self.top_k = top_k
        self.number_of_items = number_of_items
        message = (
            f"Provided top_k parameter ({top_k}) "
            f"cannot be larger than the number of items ({number_of_items})"
        )
        super().__init__(message)
