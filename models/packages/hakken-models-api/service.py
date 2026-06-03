import sys

from spaice_inference_api import Settings, create_server

from hakken_models_api.config import get_config
from hakken_models_api.container import Container
from hakken_models_api.loaders import get_loader
from hakken_models_api.routers import get_router


def main() -> None:
    api_config = get_config()

    api_config.param_overrides.update(
        {
            "segal/embedder/model_name": "mxbai-embed-large:335m",
        }
    )

    container = Container(config=api_config)
    container.init_resources()
    container.wire()
    container.wire(modules=[__name__])

    model_loader = get_loader(api_config.model)
    routers = [(None, get_router(api_config.model))]

    server = create_server(
        model_loader=model_loader,
        routers=routers,
        setup_ml_framework=None,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["hakken_models_api", "hakken_models"],
        },
        settings=Settings(),
    )

    server.run()


if __name__ == "__main__":
    main()
