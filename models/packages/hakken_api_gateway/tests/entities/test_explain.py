import pytest
from pydantic import ValidationError

from hakken_api_gateway.core.entities.explain import (
    ExplainReqFromUpstream,
    ExplainReqToDownstream,
    ExplainResp,
    ExplanationConfig,
    ExplanationItem,
)

# ==========================================
# TEST SUITE
# ==========================================


# 1. Tests for ExplainTripleRequestFromUI
# ---------------------------------------------------------
def test_ui_request_alias_parsing():
    """Test that input JSON using camelCase 'estimatedTime' maps to snake_case 'estimated_time'."""
    data = {
        "triple": ["subject", "predicate", "object"],
        "estimatedTime": 5000,  # Input uses the alias
    }
    model = ExplainReqFromUpstream(**data)

    assert model.triple == ["subject", "predicate", "object"]
    assert model.estimated_time == 5000


def test_ui_request_python_naming():
    """Test that we can also instantiate using the python attribute name
    (if allow_population_by_field_name is default on v2 or configured)."""

    model = ExplainReqFromUpstream(triple=["a", "b", "c"], estimatedTime=100)

    assert model.estimated_time == 100


def test_ui_request_validation_error():
    """Test failure when 'triple' is not a list."""
    data = {"triple": "not-a-list", "estimatedTime": 100}
    with pytest.raises(ValidationError) as exc:
        ExplainReqFromUpstream(**data)
    assert exc.value.errors()[0]["loc"][0] == "triple"


# 2. Tests for ExplanationConfig
# ---------------------------------------------------------
def test_config_defaults():
    """Test that defaults (batch_size=32, type='sufficient') are applied."""
    model = ExplanationConfig()
    assert model.batch_size == 32
    assert model.type == "sufficient"


def test_config_custom_values():
    """Test overriding defaults."""
    model = ExplanationConfig(batch_size=64, type="necessary")
    assert model.batch_size == 64
    assert model.type == "necessary"


# 3. Tests for ExplainTripleRequestToExplainer
# ---------------------------------------------------------
def test_explainer_request_nested_structure():
    """Test the nested structure of lists and config objects."""
    config_data = {"batch_size": 10, "type": "all"}
    data = {
        "triples_to_probe": [["s", "p", "o"], ["s2", "p2", "o2"]],
        "num_explanations": 5,
        "explanation_configs": [config_data],
    }

    model = ExplainReqToDownstream(**data)

    assert len(model.triples_to_probe) == 2
    assert model.triples_to_probe[0] == ["s", "p", "o"]
    assert model.num_explanations == 5
    assert len(model.explanation_configs) == 1
    assert isinstance(model.explanation_configs[0], ExplanationConfig)
    assert model.explanation_configs[0].batch_size == 10


def test_explainer_request_defaults():
    """Test that num_explanations defaults to 10."""
    data = {"triples_to_probe": [["s", "p", "o"]], "explanation_configs": []}
    model = ExplainReqToDownstream(**data)
    assert model.num_explanations == 10


def test_explainer_request_invalid_list_structure():
    """Test validation fails if triples_to_probe is not a list of lists."""
    data = {
        "triples_to_probe": ["s", "p", "o"],  # Wrong: This is a 1D list, expects 2D
        "explanation_configs": [],
    }
    with pytest.raises(ValidationError) as exc:
        ExplainReqToDownstream(**data)
    # Pydantic will complain that the inner items are strings, not lists
    assert "triples_to_probe" in str(exc.value)


# 4. Tests for ExplanationItem
# ---------------------------------------------------------
def test_explanation_item_types():
    """Test simple scalar types."""
    data = {"data": "path-string", "length": 3, "score": 0.95}
    model = ExplanationItem(**data)
    assert model.score == 0.95
    assert isinstance(model.score, float)


def test_explanation_item_casting():
    """Test that strings looking like numbers are cast to numbers."""
    data = {"data": "path", "length": "5", "score": "0.1"}
    model = ExplanationItem(**data)
    assert model.length == 5
    assert model.score == 0.1


# 5. Tests for ExplainTripleResponse
# ---------------------------------------------------------
def test_response_dynamic_dict_keys():
    """Test the dict structure with dynamic string keys."""
    item_data = {"data": "path1", "length": 2, "score": 0.5}
    data = {"explanations": {"dynamic_key_1": [item_data], "dynamic_key_2": []}}

    model = ExplainResp(**data)

    assert "dynamic_key_1" in model.explanations
    assert len(model.explanations["dynamic_key_1"]) == 1
    assert isinstance(model.explanations["dynamic_key_1"][0], ExplanationItem)
    assert model.explanations["dynamic_key_1"][0].data == "path1"


def test_response_invalid_value_structure():
    """Test failure if the value in the dict is not a list of items."""
    data = {"explanations": {"bad_key": "this should be a list of objects"}}
    with pytest.raises(ValidationError):
        ExplainResp(**data)
