from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextualization.core.entities.publication import PublicationId


class PublicationNotFoundError(Exception):
    def __init__(self, publication_id: "PublicationId"):
        self.publication_id = publication_id

    def __str__(self):
        return f"Publication not found for publication ID {self.publication_id}"


class DatabaseValidationError(Exception):
    pass


class ConfigurationError(Exception):
    pass


class RetrievalWarning(UserWarning):
    pass


class InitializationError(Exception):
    pass
