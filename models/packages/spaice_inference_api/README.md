# Spaice Inference API

This package is a base package to be used by other modules so that they can expose interfaces through REST APIs
without having to re-implement custom logging, metrics and other monitoring tools.
It includes:
- grafana metrics
- loki logs
- FASTApi server with the ability to add routers
- Interfaces so that modules can define their own way of loading a model if should be included

## Tensorflow consumption

TBD in case we need to provide setup actions for gpu mem or other actions on startup relevant to tf framework

## Torch consumption

TBD in case we need to provide setup actions for gpu mem or other actions on startup relevant to pytorch framework


### Includes
- Server with FASTApi
- Custom features and metrics to monitor with grafana and [prometheus](https://prometheus.io/docs/visualization/grafana/)
- Custom logger with labels for Logging with Loki [grafana](https://grafana.com/oss/loki/)

#### Sample service
Inside the folder but not included in the pip package, there is a sample service folder.
You can consider this package as the package (some utility or model) you are developing.
The service folder is a sample of the **the extra part** you would have to do in case you have a model to develop
- provide a way to load your model -> This is optional if you do not have a model skip
- add extra router endpoints to the server
- any other relevant implementation of yours

App settings:
PORT and HOST is info about where your server is running.
You should make sure to update:
    SPAICE_MODEL_NAME to your model name
    SPAICE_MODEL_PATH to the path where your model is after you download it, or you can use
        runtime code to download your model in the load model method
    SPAICE_INFERENCE_API_KEY: Optional[str] = None
        this is in case you need to add API token authorization to your service

There is a service.py file. By this you define your runtime file. You use the create_server function to startup the
server and provide as arguments all the extra it needs.

>Readme to be updated based on input of the 1st person to integrate this package and include any info missing

### How to update the package
This package is built with UV. Having said that when you want to develop do:
1. `uv sync --index-strategy unsafe-best-match` ->  This creates an environment dedicated to this project/package
2. after that run every command with `uv run` suffixed. Then everything runs
 with this env activated

Before committing make sure to run the following so that the CI passes:
Please run:
>uv run mypy && flake8 && isort . && pytest

### Run the sample service
`uv run -m service.app`

### How to integrate with a model
1. Install the package (inference_api) as a dependency
2. The service `service_sample` folder works like the model-service or the package of your module.
It can contain any other folder or structure. We prefer the clean architecture defined inside the team.
3. In your `service.app` module or `service.py` module (outside your working package) you will have to 
include your own blueprints/routers so that you expose your REST endpoint to provide the model predictions.


### How to utilize exception decorator

There is implemented a decorator called `wrapped_error`

Please visit the links commented on the method definition to 
familiarize with the purpose and what is doing

Arguments:
- `exception`: the custom exception to throw and wrap for the method that wraps
- `msg`: the custom msg to show
- `allowed_exceptions`: the allowed exceptions to rethrow as is without wrapping them 
   with the custom exception given