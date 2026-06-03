class GraphNotLoadedError(Exception):
    """Raised when knowledge graph is not yet loaded."""

    pass


class LoadNotImplementedError(Exception):
    """Raised when load method is not implemented in the derived class."""

    pass


class NotInitializedError(Exception):
    """Raised when an object is not initialized."""

    pass


class SplitsError(Exception):
    """Raised when list of splits not provided."""

    pass


class WrongCorruptionSchemeError(Exception):
    pass


class WrongDimensionsError(Exception):
    pass
