import dataclasses
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from spaice_inference_api.core.action.health_check import HealthCheckAction
    from spaice_inference_api.core.action.load_model import LoadModelAction

DispatcherToken = "dispatcher"


class DispatcherActions(TypedDict):
    HealthCheckAction: "HealthCheckAction"
    LoadModelAction: "LoadModelAction"


@dataclasses.dataclass
class Dispatcher:
    actions: DispatcherActions
