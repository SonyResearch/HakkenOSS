class UnknownReductionError(Exception):
    """Raised when the reduction method is unknown."""

    pass


class ShapeMismatchError(Exception):
    """Raised when the positive_scores shape is invalid."""

    pass
