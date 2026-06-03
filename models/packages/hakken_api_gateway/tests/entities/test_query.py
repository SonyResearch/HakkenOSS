import pytest
from pydantic import ValidationError

from hakken_api_gateway.core.entities.query import QueryReqFromUpstream


@pytest.fixture
def valid_json_data():
    """Returns the valid JSON dictionary from the prompt."""
    return {
        "queryApi": {
            "variables": [{"label": "X", "domain": "DISEASE"}],
            "formula": "P(X, ASSOCIATE, 43423423423)",
        },
        "queryString": "[X ∈ DISEASE]: P(X, ASSOCIATE, Parkinson Disease)",
        "hypotheses": {
            "0": {
                "condition": {
                    "head": {
                        "isVariable": True,
                        "label": "X",
                        "domain": "DISEASE",
                        "id": "DISEASE",
                    },
                    "tail": {
                        "isVariable": False,
                        "label": "Parkinson Disease",
                        "domain": "GENE",
                        "id": "43423423423",
                    },
                    "relation": "ASSOCIATE",
                },
                "addValue": "AND",
                "conditionType": "P",
            }
        },
        "constraints": {},
        "candidatesNumber": 6,
        "queryMode": "simple",
    }


def test_valid_model_parsing(valid_json_data):
    """Test that valid JSON parses correctly into the Pydantic model."""
    model = QueryReqFromUpstream.model_validate(valid_json_data)

    # Check top level simple fields
    assert model.query_string == "[X ∈ DISEASE]: P(X, ASSOCIATE, Parkinson Disease)"
    assert model.query_mode == "simple"


def test_alias_mapping(valid_json_data):
    """Test that camelCase JSON fields map correctly to snake_case attributes."""
    model = QueryReqFromUpstream.model_validate(valid_json_data)

    # candidatesNumber -> candidates_number
    assert model.candidates_number == 6

    # queryApi -> query_api
    assert len(model.query_api.variables) == 1
    assert model.query_api.variables[0].domain == "DISEASE"


def test_nested_alias_and_id_mapping(valid_json_data):
    """Test deeply nested objects and specific _id -> id mapping."""
    model = QueryReqFromUpstream.model_validate(valid_json_data)

    hypothesis = model.hypotheses["0"]

    # Check camelCase inside nested object (addValue -> add_value)
    assert hypothesis.add_value == "AND"

    # Check nested GraphNode alias (isVariable -> is_variable)
    assert hypothesis.condition.head.is_variable is True
    assert hypothesis.condition.tail.is_variable is False


def test_missing_required_field(valid_json_data):
    """Test that missing a required field raises a ValidationError."""
    # Remove a required field
    del valid_json_data["queryApi"]

    with pytest.raises(ValidationError) as excinfo:
        QueryReqFromUpstream.model_validate(valid_json_data)

    # Verify the error is about the missing field
    errors = excinfo.value.errors()
    assert any(e["loc"] == ("queryApi",) and e["type"] == "missing" for e in errors)


def test_invalid_type_conversion(valid_json_data):
    """Test that providing the wrong data type raises a ValidationError."""
    # candidatesNumber expects int, pass a non-numeric string
    valid_json_data["candidatesNumber"] = "six"

    with pytest.raises(ValidationError) as excinfo:
        QueryReqFromUpstream.model_validate(valid_json_data)

    errors = excinfo.value.errors()
    assert any(e["loc"] == ("candidatesNumber",) for e in errors)


def test_populate_by_name(valid_json_data):
    """Test that we can also initialize using the pythonic names directly."""
    # Since we set populate_by_name=True, we should be able to pass 'candidates_number'
    # instead of 'candidatesNumber' if we were constructing the dict manually.
    data = valid_json_data.copy()
    del data["candidatesNumber"]
    data["candidates_number"] = 10

    model = QueryReqFromUpstream.model_validate(data)
    assert model.candidates_number == 10
