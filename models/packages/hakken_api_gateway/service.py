import sys

from dotenv import load_dotenv
from spaice_inference_api import Settings, create_server

from hakken_api_gateway.router import router

load_dotenv(override=False)


def main() -> None:
    server = create_server(
        model_loader=None,
        routers=[(None, router)],
        setup_ml_framework=None,
        wiring_config={
            "modules": [sys.modules[__name__]],
            "packages": ["hakken_api_gateway"],
        },
        settings=Settings(SPAICE_MODEL_NAME="gateway"),
    )

    server.run()


if __name__ == "__main__":
    main()
