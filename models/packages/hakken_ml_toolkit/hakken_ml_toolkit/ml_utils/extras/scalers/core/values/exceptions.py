class ScalerNotFittedError(Exception):
    """Exception raised when a model or transformer is used before being fitted."""

    def __init__(self):
        self.message = """
        Scaler is not fitted yet. Call 'fit' with appropriate 
        inputs before using this method."""
        super().__init__(self.message)


class ZeroRangeError(Exception):
    """Exception raised when feature scaling is attempted on data with zero range."""

    def __init__(self, message="Data range is 0 for some of the features. Cannot scale."):
        self.message = message
        super().__init__(self.message)


class ZeroStandardDeviationError(Exception):
    """Exception raised when scaling is attempted on features with zero standard deviation."""

    def __init__(self, message="Standard deviation is 0 for some of the features. Cannot scale."):
        self.message = message
        super().__init__(self.message)
