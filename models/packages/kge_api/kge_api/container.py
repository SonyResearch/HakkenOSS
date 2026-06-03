from dependency_injector import containers, providers
from dotenv import load_dotenv

from kge_api.config import APIConfig

load_dotenv()


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["kge_api", "datasets"])
    # The following is so that we can pass the container itself
    # as a dependency
    __self__: providers.Self["Container"] = providers.Self()

    config = providers.Dependency(APIConfig)
