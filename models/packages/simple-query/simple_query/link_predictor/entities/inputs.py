from typing import TYPE_CHECKING

from query_common.entities.kg.triple import Triple

from simple_query.link_predictor.values.errors import LinkPredictorInputTripleError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simple_query.query.entities.inputs import Predicate


LinkPredictorInputTriple = Triple
"""Defines the input triple to be used as input to a link predictor.
Note that, we use node IDs for subject and object, and use type name for relation."
"""


def convert_predicate_to_link_predictor_input_triple(
    predicate: "Predicate", variable_substitution: "Mapping[str, str] | None" = None
) -> LinkPredictorInputTriple:
    """
    Converts a `Predicate` object used in querying
    into `LinkPredictorInputTriple` used in link prediction.

    Args:
        predicate: `Predicate` object.
        variable_substitution:
            Mapping from a variable to a node ID.
            This is used when we want to replace a variable (e.g. `X`)
            into a specific value (e.g. some node ID).
    """
    if variable_substitution:
        predicate = predicate.model_copy(deep=True)
        for argument in (predicate.subject, predicate.object):
            if argument.is_variable and argument.value in variable_substitution:
                argument.is_variable = False
                argument.value = variable_substitution[argument.value]

    if (
        predicate.subject.is_variable
        or predicate.relation.is_variable
        or predicate.object.is_variable
    ):
        raise LinkPredictorInputTripleError(
            f"variables not subtituted exists in the predicate: {predicate}"
        )

    return LinkPredictorInputTriple(
        subject_identifier=predicate.subject.value,
        relation_identifier=predicate.relation.value,
        object_identifier=predicate.object.value,
    )
