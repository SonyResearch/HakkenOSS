from datetime import datetime

import pytest
from pydantic import ValidationError

from hakken_api_gateway.core.entities.user import UserDBModel


def test_user_model_valid_instantiation():
    """Test creating a user with valid python types."""
    now = datetime.now()

    data = {
        "email": "jane@example.com",
        "name": "Jane Doe",
        "created_at": now,
        "updated_at": now,
        "last_login_at": now,
    }

    user = UserDBModel(**data)

    assert user.email == "jane@example.com"
    assert user.name == "Jane Doe"
    assert user.created_at == now
    assert isinstance(user.created_at, datetime)


def test_user_model_datetime_coercion():
    """Test that Pydantic automatically parses ISO date strings into datetime objects."""
    data = {
        "email": "john@example.com",
        "name": "John Doe",
        # Passing strings instead of datetime objects
        "created_at": "2023-10-01T12:00:00",
        "updated_at": "2023-10-02T15:30:00",
        "last_login_at": "2023-10-05T09:00:00",
    }

    user = UserDBModel(**data)

    # Assert they were converted to datetime objects
    assert isinstance(user.created_at, datetime)
    assert user.created_at.year == 2023
    assert user.created_at.month == 10


def test_user_model_missing_field():
    """Test that validation fails if a required field is missing."""
    data = {
        "email": "incomplete@example.com",
        "name": "Incomplete User",
        "created_at": datetime.now(),
        # "updated_at" and "last_login_at" are MISSING
    }

    with pytest.raises(ValidationError) as exc_info:
        UserDBModel(**data)

    # Check that the error contains the names of the missing fields
    errors = exc_info.value.errors()
    missing_fields = [err["loc"][0] for err in errors]

    assert "updated_at" in missing_fields


def test_user_model_invalid_datetime():
    """Test that validation fails if a date string is malformed."""
    data = {
        "email": "test@example.com",
        "name": "Test",
        "created_at": "not-a-date",  # Invalid
        "updated_at": datetime.now(),
        "last_login_at": datetime.now(),
    }

    with pytest.raises(ValidationError) as exc_info:
        UserDBModel(**data)

    assert exc_info.value.errors()[0]["type"] == "datetime_from_date_parsing"


def test_user_model_invalid_types():
    """Test that validation fails if we pass a dict where a string is expected."""
    data = {
        "email": {"not": "a string"},  # Invalid
        "name": "Valid Name",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "last_login_at": datetime.now(),
    }

    with pytest.raises(ValidationError) as exc_info:
        UserDBModel(**data)

    assert exc_info.value.errors()[0]["loc"][0] == "email"
    assert exc_info.value.errors()[0]["type"] == "string_type"
