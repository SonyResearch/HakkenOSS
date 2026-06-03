from typing import TYPE_CHECKING

from query_common.entities.grounded_query import GroundedQuery
from query_common.entities.variable import Variable
from query_common.grounding.query_grounder import QueryGrounder

if TYPE_CHECKING:
    from query_common.entities.clauses.query import Query


def ground_query(query: "Query") -> GroundedQuery:
    variables = {
        p.variable.value: Variable(label=p.variable.value.value, domain_identifier=p.domain.value)
        for p in query.patterns
    }
    query_grounder = QueryGrounder(variables)
    condition = query_grounder.convert_formula_to_condition(query.condition)
    return GroundedQuery(variables, condition)


def ground_query_given_variables(query: "Query", variables: list[Variable]) -> GroundedQuery:
    variables_as_dict = {v.label: v for v in variables}
    query_grounder = QueryGrounder(variables_as_dict)
    condition = query_grounder.convert_formula_to_condition(query.condition)
    return GroundedQuery(variables_as_dict, condition)
