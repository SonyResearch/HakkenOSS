from langchain_core.documents import Document
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The name or identifier of the entity")
    domain: str = Field(
        description="The domain or category that this entity belongs to. Can include subdomains in hierarchical format: domain/subdomain/subsubdomain/..."
    )
    description: str = Field(
        description=(
            "A single-sentence, context-independent definition of what the entity is. "
            "Describe the entity in general knowledge terms, not how it appears or is "
            "used in the source text."
        )
    )

    def to_string(self) -> str:
        return f"{self.name} ({self.domain}):\n{self.description}"

    def metadata(self) -> dict:
        return {
            "domain": self.domain,
            "name": self.name,
            "description": self.description,
        }

    def to_document(self) -> Document:
        return Document(
            page_content=self.to_string(),
            metadata=self.metadata(),
        )


class EntityDB(Entity):
    uuid: str | None = Field(default=None, description="The UUID of the entity")
    domain_uuid: str | None = Field(default=None, description="The UUID of the domain")

    def metadata(self) -> dict:
        metadata = super().metadata()
        if self.domain_uuid is not None:
            metadata["domain_id"] = self.domain_uuid
        return metadata

    def equal(self, other: "EntityDB") -> bool:
        return self.name == other.name and self.domain == other.domain

    def equal_domain(self, other: "EntityDB") -> bool:
        if self.domain_uuid is None:
            raise ValueError("Domain UUID is not set")
        return self.domain == other.domain

    def set_uuid(self, uuid: str) -> None:
        self.uuid = uuid

    def set_from_other(self, other: "EntityDB") -> None:
        self.uuid = other.uuid
        self.domain_uuid = other.domain_uuid
        self.name = other.name
        self.domain = other.domain
        self.description = other.description

    def set_domain_info(self, name: str, uuid: str) -> None:
        self.domain = name
        self.domain_uuid = uuid

    def to_document(self) -> Document:
        data = {
            "page_content": self.to_string(),
            "metadata": self.metadata(),
        }
        if self.uuid is not None:
            data["id"] = self.uuid
        return Document(**data)

    @classmethod
    def from_document(cls, document: Document) -> "EntityDB":
        return cls(
            uuid=document.id,
            domain_uuid=document.metadata.get("domain_id"),
            name=document.metadata.get("name"),
            domain=document.metadata.get("domain"),
            description=document.metadata.get("description"),
        )

    @classmethod
    def from_entity(cls, entity: Entity) -> "EntityDB":
        return cls(
            uuid=None,
            domain_uuid=None,
            name=entity.name,
            domain=entity.domain,
            description=entity.description,
        )

    @classmethod
    def from_entity_list(cls, entities: list[Entity]) -> list["EntityDB"]:
        return [cls.from_entity(entity) for entity in entities]
