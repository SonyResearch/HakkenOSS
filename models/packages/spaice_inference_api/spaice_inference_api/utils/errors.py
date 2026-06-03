from typing import TYPE_CHECKING, TypeVar

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


# The following technique is so that the wrapper does not hide the
# typing information of the wrapped function
# See https://stackoverflow.com/questions/47060133/python-3-type-hinting-for-decorator
# and https://rednafi.github.io/reflections/static-typing-python-decorators.html
def wrapped_error(
    exception: type[Exception],
    msg: str = "Operation failed",
    allowed_exceptions: list[type[Exception]] | None = None,
) -> "Callable[[Callable[P, R]], Callable[P, R]]":
    if allowed_exceptions is None:
        allowed_exceptions = []

    def decorator(func: "Callable[P, R]") -> "Callable[P, R]":
        def wrapper(*args, **kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if type(e) in allowed_exceptions:
                    raise
                raise exception(msg) from e

        return wrapper

    return decorator
