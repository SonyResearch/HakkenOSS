from hakken_agents.tools.info_extractor import InfoExtractor

from .config import EntityExtractorConfig
from .schemas import ExtractedEntities


class EntityExtractor(InfoExtractor[ExtractedEntities, EntityExtractorConfig]):
    output_schema = ExtractedEntities

    @property
    def allowed_user_variables(self) -> list[str]:
        return ["relevant_domains", "allowed_domains", "previous_text"]

    @property
    def user_variables_are_required(self) -> bool:
        return False
