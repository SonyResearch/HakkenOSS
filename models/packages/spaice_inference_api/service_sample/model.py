import time

from dependency_injector.wiring import Provide, inject

from service_sample.entities import MyInferenceRequest, MyInferenceResponse
from spaice_inference_api import ILogger, LoggerToken, time_model_prediction


class MyModel:
    @time_model_prediction
    @inject
    def predict(
        self, _request: MyInferenceRequest, logger: ILogger = Provide[LoggerToken]
    ) -> MyInferenceResponse:
        logger.info("sleeping")
        time.sleep(1)
        logger.info("sleeping")
        time.sleep(1)
        logger.info("sleeping")
        time.sleep(1)
        logger.info("sleeping")
        time.sleep(1)
        logger.info("sleeping")
        time.sleep(1)
        logger.info("sleeping")
        time.sleep(1)
        return MyInferenceResponse(hello="world")
