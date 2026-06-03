from typing import TYPE_CHECKING

from simple_query.query.entities.configs import SimpleQueryingConfig
from simple_query.query.impl.simple import SimpleQuerying

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simple_query.query.base import Querying
    from simple_query.query.entities.configs import QueryingConfigBase


CLASS_MAPPING: "Mapping[type[QueryingConfigBase], type[Querying]]" = {
    SimpleQueryingConfig: SimpleQuerying
}
