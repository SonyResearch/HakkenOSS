from dependency_injector import containers, providers
from dotenv import load_dotenv

from hakken_models_api.config import HakkenModelsAPIConfig

load_dotenv()


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["hakken_models_api", "hakken_models"])
    # The following is so that we can pass the container itself
    # as a dependency
    __self__: providers.Self["Container"] = providers.Self()

    config = providers.Dependency(HakkenModelsAPIConfig)
