from typing import TYPE_CHECKING

import pytest
from query_common.entities.kg.triple import Triple

from complex_query.core.entities.config.score_ledger import InMemoryScoreLedgerConfig
from complex_query.core.values.class_mapping import SCORE_LEDGER_CLASS_MAPPING

if TYPE_CHECKING:
    from complex_query.core.contracts.score_ledger import ScoreLedger


@pytest.fixture(params=[InMemoryScoreLedgerConfig])
def ledger(request: pytest.FixtureRequest):
    config = request.param()
    ledger = SCORE_LEDGER_CLASS_MAPPING[type(config)](config)
    ledger.save_link_score(
        Triple(subject_identifier="n1", relation_identifier="r1", object_identifier="n3"), 0.6
    )
    ledger.save_link_score(
        Triple(subject_identifier="n1", relation_identifier="r4", object_identifier="n2"), 0.3
    )
    ledger.save_link_score(
        Triple(subject_identifier="n2", relation_identifier="r2", object_identifier="n1"), 0.0
    )
    return ledger


def test_retrieve_score(ledger: "ScoreLedger"):
    assert (
        ledger.retrieve_link_score(
            Triple(subject_identifier="n1", relation_identifier="r1", object_identifier="n3")
        )
        == 0.6
    )
    assert (
        ledger.retrieve_link_score(
            Triple(subject_identifier="n1", relation_identifier="r4", object_identifier="n2")
        )
        == 0.3
    )
    assert (
        ledger.retrieve_link_score(
            Triple(subject_identifier="n2", relation_identifier="r2", object_identifier="n1")
        )
        == 0.0
    )
    with pytest.raises(KeyError):
        ledger.retrieve_link_score(
            Triple(
                subject_identifier="n1000", relation_identifier="r1000", object_identifier="n1000"
            )
        )
