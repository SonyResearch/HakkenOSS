"""FastAPI dependencies for the Element Resolver API."""

from fastapi import Request

from hakken_agents.tools.element_resolver import ElementResolver


def get_element_resolver(request: Request) -> ElementResolver:
    """Return the ElementResolver instance from app state."""
    resolver: ElementResolver | None = getattr(request.app.state, "element_resolver", None)
    if resolver is None:
        raise RuntimeError("ElementResolver not initialized")
    return resolver
