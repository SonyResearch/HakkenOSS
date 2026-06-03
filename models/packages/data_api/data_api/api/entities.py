from pydantic import BaseModel, Field


class IdToNameRequest(BaseModel):
    id_list: list[str] = Field(..., description="List of IDs to retrieve names for")


class IdToNameResponse(BaseModel):
    id_name_mapping: dict = Field(..., description="Mapping id to name")


class UniqueDomainsResponse(BaseModel):
    domain_names: list[str] = Field(..., description="List of unique domain names")


class NodesFromDomainRequest(BaseModel):
    domain_name: str = Field(..., description="Domain name to retrieve node IDs for")
    node_name: str | None = Field(
        None, description="Optional node name filter to narrow down the results"
    )
    max_results: int = Field(500, description="Maximum number of results to return")


class NodeInfo(BaseModel):
    id: str = Field(..., description="Node ID")
    name: str = Field(..., description="Node Name")


class NodesFromDomainResponse(BaseModel):
    nodes: list[NodeInfo] = Field(..., description="List of node IDs for the given domain name")


class EdgeTypesFromNodeDomainsRequest(BaseModel):
    subject_domain: str | None = Field(None, description="Subject domain name")
    object_domain: str | None = Field(None, description="Object domain name")


class EdgeTypesFromNodeDomainsResponse(BaseModel):
    edge_types: list[str] = Field(..., description="List of edge types for the given domain names")


class NodeDomainsFromNodeDomainsAndEdgeTypeRequest(BaseModel):
    subject_domain: str | None = Field(None, description="Subject domain name")
    object_domain: str | None = Field(None, description="Object domain name")
    edge_type: str | None = Field(None, description="Edge type name")


class NodeDomainsFromNodeDomainsAndEdgeTypeResponse(BaseModel):
    node_domains: list[str] = Field(
        ..., description="List of node domains for the given domain names and edge type"
    )
