from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import hydra
from dependency_injector.wiring import Provide, inject
from spaice_inference_api import ILogger, IModelLoader, LoggerToken, ModelLoadingOptions

from simple_xkgc_api.container import Container

if TYPE_CHECKING:
    from simple_xkgc_api.entities.config import APIConfig


class PathFinder(Protocol):
    pass


class PathExplainer(Protocol):
    def setup(self, *, path_finder: PathFinder) -> None: ...


class PathExplainerLoader(IModelLoader):
    @inject
    def load(
        self,
        _options: ModelLoadingOptions,
        logger: ILogger = Provide[LoggerToken],
        config: APIConfig = Provide[Container.config],
    ) -> PathExplainer:
        logger.info(f"CONFIG:\n'{config}'")

        path_finder = cast("PathFinder", hydra.utils.instantiate(config.path_finder))

        explainer = cast("PathExplainer", hydra.utils.instantiate(config.explainer))
        explainer.setup(path_finder=path_finder)

        return explainer
