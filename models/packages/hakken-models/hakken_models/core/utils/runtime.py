"""Utilities for script."""

import ast
import importlib
from typing import Any

import typer


def instantiate_from_string(
    class_path: str, *args: Any, expected_type: type | None = None, **kwargs: Any
) -> Any:
    """Instantiate a class from a full module path string.

    Args:
        class_path: Full module path to the class (e.g., "hakken_models.models.thiger.THiGER")
        *args: Positional arguments for class constructor
        **kwargs: Keyword arguments for class constructor

    Returns:
        Instance of the class

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class is not found in the module

    Example:
        >>> instance = instantiate_from_string(
        ...     "hakken_models.models.thiger.THiGER",
        ...     num_entities=100,
        ...     num_relations=50
        ... )
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)

    if expected_type is not None and not isinstance(instance, expected_type):
        msg = (
            f"Object at '{class_path}' does not implement required type "
            f"'{expected_type.__name__}'. "
            f"Got type: {type(instance).__name__}"
        )
        raise TypeError(msg)

    return instance


def parse_override(override: str | None) -> list[str] | None:
    if override is None:
        return None

    override = override.strip()

    # Dict-style override
    if override.startswith("{"):
        try:
            override_dict = ast.literal_eval(override)

            if not isinstance(override_dict, dict):
                raise ValueError("Override must evaluate to a dict.")

            return [f"{k}={v}" for k, v in override_dict.items()]

        except (ValueError, SyntaxError) as e:
            raise typer.BadParameter(f"Invalid override dict: {e}")

    # Space-separated key=value pairs
    return override.split()


def flat_overrides_from_override_list(override_list: list[str] | None) -> dict[str, str]:
    """
    Convert CLI override tokens (as returned by :func:`parse_override`) into a flat
    ``dict`` suitable for merging onto MLflow run params before unflattening.

    Each token must contain ``=``; the key is everything before the first ``=``.
    """
    if not override_list:
        return {}
    out: dict[str, str] = {}
    for item in override_list:
        if "=" not in item:
            raise typer.BadParameter(f"Override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out
