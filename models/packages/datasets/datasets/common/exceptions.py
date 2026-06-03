class GraphNotLoadedError(Exception):
    """Raised when knowledge graph is not yet loaded."""

    pass


class InvalidDateTypeError(Exception):
    pass


class SplitNotFoundError(Exception):
    """Raised when split not found in knowledge graph."""

    pass


class KnowledgeGraphObjectError(Exception):
    """Raised when Something went wrong. Expected KnowledgeGraph object."""

    pass


class DataSplitProportionError(Exception):
    """Exception raised when the sum of data_split_proportion_dict values is not equal to 1.0."""

    def __init__(self, message="Sum of data_split_proportion_dict values must be 1.0."):
        self.message = message
        super().__init__(self.message)
