import pytest
from pydantic import ValidationError

from hakken_api_gateway.core.entities.query import (
    Candidate,
    QueryReqToDownstream,
    QueryRespFromDownstream,
    QueryVariable,
    SearchParameters,
)


# 1. SearchParameters Tests
# ----------------------------------------------------------------
def test_search_parameters_valid():
    """Test valid instantiation of SearchParameters."""
    model = SearchParameters(beam_size=5)
    assert model.beam_size == 5


def test_search_parameters_validation_error():
    """Test failure when beam_size is not an integer."""
    with pytest.raises(ValidationError) as exc:
        SearchParameters(beam_size="not-an-int")
    assert exc.value.errors()[0]["loc"][0] == "beam_size"


# 2. QueryModuleRequest Tests
# ----------------------------------------------------------------
def test_query_request_defaults():
    """Test that search_algorithm defaults to 'beam' if omitted."""
    data = {
        "formula": "X and Y",
        "variables": [{"label": "A", "domain": "int"}, {"label": "B", "domain": "str"}],
        "n_candidates": 10,
        "search_parameters": {"beam_size": 3},
    }
    model = QueryReqToDownstream(**data)
    assert model.search_algorithm == "beam"
    # Verify nested model was created
    assert isinstance(model.search_parameters, SearchParameters)
    assert model.search_parameters.beam_size == 3


def test_query_request_nested_validation():
    """Test that validation fails if the NESTED model data is invalid."""
    data = {
        "formula": "X and Y",
        "variables": [],
        "n_candidates": 10,
        # Invalid: beam_size expects int, got invalid string
        "search_parameters": {"beam_size": "bad-value"},
    }
    with pytest.raises(ValidationError) as exc:
        QueryReqToDownstream(**data)

    # Check that the error location path indicates the nested field
    # loc should look like ('search_parameters', 'beam_size')
    assert "search_parameters" in str(exc.value)
    assert "beam_size" in str(exc.value)


def test_query_request_variables_list():
    """Test that the variables list parses correctly into QueryVariable objects."""
    data = {
        "formula": "A",
        "variables": [{"label": "A", "domain": "int"}, {"label": "B", "domain": "str"}],
        "n_candidates": 1,
        "search_parameters": {"beam_size": 1},
    }
    model = QueryReqToDownstream(**data)
    assert len(model.variables) == 2
    assert model.variables[0].label == "A"
    assert isinstance(model.variables[1], QueryVariable)


# 3. Candidate Tests
# ----------------------------------------------------------------
def test_candidate_float_parsing():
    """Test that dictionaries parse correctly and floats are coerced from strings."""
    data = {
        "var_assignments": {"X": "ent_1"},
        "condition_scores": {"cond_1": "0.99", "cond_2": 0.5},  # Mixed string/float input
        "query_score": "1.0",
    }
    model = Candidate(**data)

    assert model.var_assignments["X"] == "ent_1"
    assert model.condition_scores["cond_1"] == 0.99  # Should be float
    assert model.query_score == 1.0


def test_candidate_missing_fields():
    """Test failure when required dictionaries are missing."""
    data = {
        "var_assignments": {"X": "ent_1"},
        # condition_scores is missing
        "query_score": 0.5,
    }
    with pytest.raises(ValidationError) as exc:
        Candidate(**data)
    assert exc.value.errors()[0]["loc"][0] == "condition_scores"


# 4. QueryModuleResponse Tests
# ----------------------------------------------------------------
def test_response_structure():
    """Test the full response structure containing a list of Candidates."""
    candidate_data = {"var_assignments": {"A": "1"}, "condition_scores": {}, "query_score": 0.1}

    data = {"candidates": [candidate_data, candidate_data]}

    model = QueryRespFromDownstream(**data)
    assert len(model.candidates) == 2
    assert isinstance(model.candidates[0], Candidate)
    assert model.candidates[0].query_score == 0.1


def test_response_empty_list():
    """Test that an empty candidate list is valid."""
    model = QueryRespFromDownstream(candidates=[])
    assert model.candidates == []
