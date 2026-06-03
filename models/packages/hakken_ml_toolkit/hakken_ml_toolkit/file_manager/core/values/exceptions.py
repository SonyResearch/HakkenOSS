class S3ManagerError(Exception):
    """Base exception class for S3Manager"""

    pass


class BucketNotFoundError(S3ManagerError):
    def __init__(self, bucket_name: str) -> None:
        super().__init__(f"Bucket '{bucket_name}' not found")


class InvalidCredentialsError(S3ManagerError):
    pass


class InvalidS3ProtocolError(S3ManagerError):
    """Exception raised when the S3 protocol is missing or invalid."""

    def __init__(self, path=None):
        self.path = path
        super().__init__(f"Invalid S3 protocol: {path}" if path else "Invalid S3 protocol")
