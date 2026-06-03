"""Lightweight wall-clock timing via context: decorator and scope.

When a "current timings" dict is set in context (e.g. at the start of a
training step), @timed and timed_scope record durations into it. When no
context is set, they are no-ops. Use for aggregating per-step timings to
log at epoch end.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TypeVar

_current_timings: ContextVar[dict[str, float] | None] = ContextVar(
    "current_timings",
    default=None,
)


def set_current_timings(timings: dict[str, float] | None) -> None:
    """Set the timings dict for the current context (e.g. start of a step)."""
    _current_timings.set(timings)


def get_current_timings() -> dict[str, float] | None:
    """Get the current timings dict, or None if not set."""
    return _current_timings.get()


F = TypeVar("F", bound=Callable[..., object])


def timed(key: str) -> Callable[[F], F]:
    """Decorator: record wall-clock duration of the call into current timings dict.

    If no timings context is set, the call runs unchanged with no recording.
    The key is stored in milliseconds (ms).
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            timings = _current_timings.get()
            if timings is None:
                return fn(*args, **kwargs)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            timings[key] = (time.perf_counter() - t0) * 1000
            return out

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timed_scope(key: str) -> object:
    """Context manager: record wall-clock duration of the block into current timings dict.

    If no timings context is set, the block runs unchanged with no recording.
    The key is stored in milliseconds (ms).
    """
    timings = _current_timings.get()
    if timings is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = (time.perf_counter() - t0) * 1000
