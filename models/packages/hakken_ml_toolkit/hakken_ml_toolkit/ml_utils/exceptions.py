class InvalidYamlError(Exception):
    """Raised when yaml file not found."""

    pass


class InvalidDictConfigError(Exception):
    """Raised when invalid DictConfig is set."""

    pass


class UnsupportedInitializationError(Exception):
    """Raised when Unsupported initialization strategy occurs."""

    pass


class DelimiterExtensionMismatchError(Exception):
    """
    Exception raised when there is a mismatch between the
    specified delimiter and file extension."""

    pass


class StratifiedSamplingError(Exception):
    """Custom exception for stratified sampling function errors."""

    pass
