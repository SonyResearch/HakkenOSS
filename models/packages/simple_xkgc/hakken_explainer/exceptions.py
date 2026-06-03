class SetupNotCalledError(Exception):
    """Raised when a method is called before setup() has been called."""

    pass


class FactGenerationError(ValueError):
    """Raised when fact generation is called with invalid or insufficient arguments."""

    pass


class MissingRequiredArgumentError(ValueError):
    """Raised when a required argument is missing from a keyword-argument setup."""

    def __init__(self, argument_name: str, message: str | None = None):
        if message is None:
            message = f"Missing required argument: '{argument_name}'."
        super().__init__(message)
        self.argument_name = argument_name
