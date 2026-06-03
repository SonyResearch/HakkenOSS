"""API routers for hakken-models-api."""

from fastapi import APIRouter
from hakken_models.core.constants import ModelType

from hakken_models_api.routers.segal import router as segal_router
from hakken_models_api.routers.thiger import router as thiger_router

ROUTER_REGISTRY: dict[ModelType, APIRouter] = {
    ModelType.SEGAL: segal_router,
    ModelType.THIGER: thiger_router,
}


def get_router(model_type: ModelType) -> APIRouter:
    """Return the router for the given model type."""
    if model_type not in ROUTER_REGISTRY:
        raise NotImplementedError(
            f"Router for model '{model_type}' is not implemented. "
            f"Available: {list(ROUTER_REGISTRY.keys())}"
        )
    return ROUTER_REGISTRY[model_type]


__all__ = ["ROUTER_REGISTRY", "get_router", "segal_router", "thiger_router"]
