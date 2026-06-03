import pytest

from data_processing.temporal_kg_engine.base import TemporalKGEngine


def test_abstract_class_cannot_be_instantiated():
    """Test that TemporalKGEngine cannot be instantiated directly."""
    with pytest.raises(TypeError):
        TemporalKGEngine()
