from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for classes with string-based lookup and type safety."""

    def __init__(self, name: str):
        """Initialize registry with a descriptive name for error messages.

        Args:
            name: Name of the registry (e.g., "Loss", "Optimizer")
        """
        self.name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class.

        Usage:
            @registry.register("my_loss")
            class MyLoss(nn.Module):
                ...

        Or register by class name:
            @registry.register()
            class MyLoss(nn.Module):
                ...
        """

        def decorator(cls: type[T]) -> type[T]:
            key = name if name is not None else cls.__name__
            if key in self._registry:
                raise ValueError(
                    f"{self.name} registry: '{key}' is already registered. "
                    f"Existing: {self._registry[key]}, New: {cls}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def register_class(self, cls: type[T], name: str | None = None) -> None:
        """Manually register a class (alternative to decorator).

        Args:
            cls: Class to register
            name: Registration key (defaults to cls.__name__ if None)

        Raises:
            ValueError: If name is already registered
        """
        key = name if name is not None else cls.__name__
        if key in self._registry:
            raise ValueError(
                f"{self.name} registry: '{key}' is already registered. "
                f"Existing: {self._registry[key]}, New: {cls}"
            )
        self._registry[key] = cls

    def get(self, name: str) -> type[T]:
        """Get a registered class by name.

        Args:
            name: Registration key

        Returns:
            Registered class

        Raises:
            ValueError: If name is not found
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(f"{self.name} registry: '{name}' not found. Available: {available}")
        return self._registry[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Get and instantiate a registered class.

        Args:
            name: Registration key
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor

        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list_all(self) -> list[str]:
        """List all registered names.

        Returns:
            Sorted list of registered keys
        """
        return sorted(self._registry.keys())

    def unregister(self, name: str) -> None:
        """Remove a registration (useful for testing).

        Args:
            name: Registration key to remove

        Raises:
            ValueError: If name is not in registry
        """
        if name not in self._registry:
            raise ValueError(f"{self.name} registry: '{name}' not in registry")
        del self._registry[name]

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
            name: Registration key to check

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry
