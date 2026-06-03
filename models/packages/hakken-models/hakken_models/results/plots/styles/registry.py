from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl

_STYLE_REGISTRY: dict[str, dict] = {}


def register_style(name: str, rcparams: dict, *, overwrite: bool = False) -> None:
    if (not overwrite) and name in _STYLE_REGISTRY:
        raise ValueError(f"Style '{name}' already registered.")
    _STYLE_REGISTRY[name] = rcparams


def list_styles() -> list[str]:
    return sorted(_STYLE_REGISTRY.keys())


def use_style(name: str) -> None:
    """Apply style globally."""
    if name not in _STYLE_REGISTRY:
        raise KeyError(f"Unknown style '{name}'. Available: {list_styles()}")
    mpl.rcParams.update(_STYLE_REGISTRY[name])


@contextmanager
def mpl_style(name: str) -> Iterator[None]:
    """Apply style temporarily (recommended)."""
    if name not in _STYLE_REGISTRY:
        raise KeyError(f"Unknown style '{name}'. Available: {list_styles()}")
    with mpl.rc_context(rc=_STYLE_REGISTRY[name]):
        yield
