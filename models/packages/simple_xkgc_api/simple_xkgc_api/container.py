from dependency_injector import containers, providers

from simple_xkgc_api.entities.config import APIConfig


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["simple_xkgc_api", "simple_xkgc"])
    # The following is so that we can pass the container itself
    # as a dependency
    __self__: providers.Self["Container"] = providers.Self()

    config = providers.Dependency(APIConfig)
