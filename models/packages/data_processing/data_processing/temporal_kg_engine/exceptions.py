from __future__ import annotations

from typing import Any


class DatabaseNotConnectedError(Exception):
    """Raised when attempting to perform database operations without being connected."""

    def __init__(self) -> None:
        super().__init__("Not connected to database. Call connect() first.")


class DataNotLoadedError(Exception):
    """Raised when required data (nodes/edges) has not been loaded before use."""

    def __init__(self) -> None:
        super().__init__("Data not loaded. Call load_data(nodes, edges) first.")


class ConfigurationError(ValueError):
    """
    Exception raised when a required configuration attribute is missing or invalid.

    Example
    -------
    >>> raise ConfigurationError.missing(
    ...     attr="local_data_storage_dir",
    ...     context={"graph_name": "my_kg"},
    ...     hint="Use a template containing '{graph_name}'."
    ... )
    """

    @classmethod
    def missing(
        cls,
        *,
        attr: str,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> ConfigurationError:
        """
        Construct an error for a missing configuration attribute.

        Parameters
        ----------
        attr : str
            Name of the missing attribute (e.g., ``local_data_storage_dir``).
        context : dict[str, Any], optional
            Additional context values to include in the message (e.g., ``graph_name``).
        hint : str, optional
            Suggestion for how to fix or configure the missing attribute.

        Returns
        -------
        ConfigurationError
            The constructed exception with a formatted error message.
        """
        parts = [f"`{attr}` is not configured."]

        if context:
            for key, value in context.items():
                parts.append(f"{key}: {value!r}")

        if hint:
            parts.append(hint)

        return cls(" ".join(parts))
