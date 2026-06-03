from spaice_inference_api.core.errors import HealthCheckError
from spaice_inference_api.utils.errors import wrapped_error


class HealthCheckResponse:
    def __init__(self, status: str):
        self.status = status


class HealthCheckAction:
    @wrapped_error(HealthCheckError)
    def go(self) -> HealthCheckResponse:
        # TODO: reconsider what we want to check here
        return HealthCheckResponse(status="ok")
