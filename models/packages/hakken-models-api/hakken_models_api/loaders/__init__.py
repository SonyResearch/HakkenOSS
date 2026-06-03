"""Model loaders for hakken-models-api."""

from hakken_models.core.constants import ModelType
from spaice_inference_api import IModelLoader

from hakken_models_api.loaders.segal import SeGALRunLoader
from hakken_models_api.loaders.thiger import THiGERRunLoader

LOADER_REGISTRY: dict[ModelType, type[IModelLoader]] = {
    ModelType.SEGAL: SeGALRunLoader,
    ModelType.THIGER: THiGERRunLoader,
}


def get_loader(model_type: ModelType) -> type[IModelLoader]:
    """Return the loader class for the given model type."""
    if model_type not in LOADER_REGISTRY:
        raise NotImplementedError(
            f"Loader for model '{model_type}' is not implemented. "
            f"Available: {list(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[model_type]


__all__ = ["LOADER_REGISTRY", "SeGALRunLoader", "THiGERRunLoader", "get_loader"]
