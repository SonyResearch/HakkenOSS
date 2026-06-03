import sys

from spaice_inference_api import Settings, create_server

from data_api.router import router


def main() -> None:
    server = create_server(
        model_loader=None,
        routers=[(None, router)],
        setup_ml_framework=None,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["data_api"],
        },
        settings=Settings(SPAICE_MODEL_NAME="core-model"),
    )
    server.run()


if __name__ == "__main__":
    main()
