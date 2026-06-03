from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from embeddings.container import Container


class WiringConfig(BaseModel):
    packages: Iterable = []
    modules: Iterable = []


def setup_container(configuration: Path | None = None, wiring_config=WiringConfig()):
    container = Container()
    if configuration is not None:
        container.config.from_yaml(configuration)
    container.init_resources()
    container.wiring_config.modules.extend(wiring_config.modules)
    container.wiring_config.packages.extend(wiring_config.packages)
    container.wire()
    return container
