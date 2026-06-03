import pytest
from pydantic import ValidationError

from data_api.api.entities import (
    EdgeTypesFromNodeDomainsRequest,
    EdgeTypesFromNodeDomainsResponse,
    IdToNameRequest,
    IdToNameResponse,
    NodeDomainsFromNodeDomainsAndEdgeTypeRequest,
    NodeDomainsFromNodeDomainsAndEdgeTypeResponse,
    NodeInfo,
    NodesFromDomainRequest,
    NodesFromDomainResponse,
    UniqueDomainsResponse,
)

# --- IdToNameRequest Tests ---


def test_id_to_name_request_valid():
    """Test valid creation of IdToNameRequest."""
    data = {"id_list": ["id1", "id2", "id3"]}
    model = IdToNameRequest(**data)
    assert model.id_list == ["id1", "id2", "id3"]


def test_id_to_name_request_missing_field():
    """Test missing required field in IdToNameRequest."""
    with pytest.raises(ValidationError) as excinfo:
        IdToNameRequest()
    assert "Field required" in str(excinfo.value)
    assert "id_list" in str(excinfo.value)


def test_id_to_name_request_invalid_type():
    """Test invalid type for id_list."""
    with pytest.raises(ValidationError):
        IdToNameRequest(id_list="not a list")


# --- IdToNameResponse Tests ---


def test_id_to_name_response_valid():
    """Test valid creation of IdToNameResponse."""
    data = {"id_name_mapping": {"id1": "name1", "id2": "name2"}}
    model = IdToNameResponse(**data)
    assert model.id_name_mapping["id1"] == "name1"


# --- UniqueDomainsResponse Tests ---


def test_unique_domains_response_valid():
    """Test valid creation of UniqueDomainsResponse."""
    data = {"domain_names": ["domain1", "domain2"]}
    model = UniqueDomainsResponse(**data)
    assert len(model.domain_names) == len(data["domain_names"])


# --- NodesFromDomainRequest Tests ---


def test_nodes_from_domain_request_defaults():
    """Test NodesFromDomainRequest defaults (max_results=500, node_name=None)."""
    max_results = 500

    model = NodesFromDomainRequest(domain_name="example.com")
    assert model.domain_name == "example.com"
    assert model.node_name is None
    assert model.max_results == max_results


def test_nodes_from_domain_request_full():
    """Test NodesFromDomainRequest with all fields provided."""
    max_results = 10

    model = NodesFromDomainRequest(domain_name="example.com", node_name="node1", max_results=10)
    assert model.node_name == "node1"
    assert model.max_results == max_results


def test_nodes_from_domain_request_missing_required():
    """Test missing domain_name."""
    with pytest.raises(ValidationError):
        NodesFromDomainRequest(node_name="node1")


# --- NodeInfo & NodesFromDomainResponse Tests ---


def test_node_info_valid():
    """Test NodeInfo creation."""
    model = NodeInfo(id="123", name="Test Node")
    assert model.id == "123"
    assert model.name == "Test Node"


def test_nodes_from_domain_response_valid():
    """Test nesting NodeInfo inside NodesFromDomainResponse."""
    nodes = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]
    model = NodesFromDomainResponse(nodes=nodes)
    assert len(model.nodes) == len(nodes)
    assert isinstance(model.nodes[0], NodeInfo)
    assert model.nodes[0].name == nodes[0]["name"]


# --- EdgeTypesFromNodeDomainsRequest Tests ---


def test_edge_types_request_defaults():
    """Test that optional fields default to None."""
    model = EdgeTypesFromNodeDomainsRequest()
    assert model.subject_domain is None
    assert model.object_domain is None


def test_edge_types_request_partial():
    """Test providing only one optional field."""
    model = EdgeTypesFromNodeDomainsRequest(subject_domain="Sales")
    assert model.subject_domain == "Sales"
    assert model.object_domain is None


# --- EdgeTypesFromNodeDomainsResponse Tests ---


def test_edge_types_response_valid():
    model = EdgeTypesFromNodeDomainsResponse(edge_types=["reports_to", "managed_by"])
    assert "reports_to" in model.edge_types


# --- NodeDomainsFromNodeDomainsAndEdgeTypeRequest Tests ---


def test_node_domains_complex_request_defaults():
    """Test default values are None."""
    model = NodeDomainsFromNodeDomainsAndEdgeTypeRequest()
    assert model.subject_domain is None
    assert model.object_domain is None
    assert model.edge_type is None


def test_node_domains_complex_request_full():
    """Test providing all fields."""
    model = NodeDomainsFromNodeDomainsAndEdgeTypeRequest(
        subject_domain="A", object_domain="B", edge_type="link"
    )
    assert model.edge_type == "link"


# --- NodeDomainsFromNodeDomainsAndEdgeTypeResponse Tests ---


def test_node_domains_complex_response_valid():
    model = NodeDomainsFromNodeDomainsAndEdgeTypeResponse(node_domains=["DomainA", "DomainB"])
    assert model.node_domains == ["DomainA", "DomainB"]


def test_node_domains_complex_response_invalid():
    """Test validation error if list is not provided."""
    with pytest.raises(ValidationError):
        NodeDomainsFromNodeDomainsAndEdgeTypeResponse(node_domains="not-a-list")
