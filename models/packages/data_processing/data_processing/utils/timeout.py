from collections.abc import Callable
from functools import wraps
from threading import Thread
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


def timeout_fn(seconds: int | None = None):
    """
    Decorator to enforce a timeout on a function.

    Args:
        seconds: Timeout in seconds. None = no timeout.

    Raises:
        TimeoutError: If function exceeds timeout.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if seconds is None or seconds <= 0:
                return func(*args, **kwargs)

            # Use a dict to store result/exception (mutable container in closure)
            state: dict[str, Any] = {"result": None, "exception": None}

            def target() -> None:
                try:
                    result = func(*args, **kwargs)
                    state["result"] = result
                    state["exception"] = None
                except Exception as e:
                    state["result"] = None
                    state["exception"] = e

            thread = Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                msg = f"Function '{func.__name__}' timed out after {seconds}s"
                raise TimeoutError(msg)

            if state["exception"] is not None:
                raise state["exception"]

            return cast("R", state["result"])

        return wrapper

    return decorator
