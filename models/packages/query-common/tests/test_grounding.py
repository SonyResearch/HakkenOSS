import pytest

from query_common.entities.variable import Variable
from query_common.grounding.actions import ground_query, ground_query_given_variables
from query_common.parse.impl.lark import LarkParser
from query_common.values.keywords import LinkPredicate

parser = LarkParser()
domain_a = "domain_a"
domain_b = "domain_b"
variable_x = Variable(label="x", domain_identifier=domain_a)
variable_y = Variable(label="b", domain_identifier=domain_b)
treats = "r12"
interacts = "r6"
assoc = "r3"
drug_a = "1056"
drug_b = "3513"
query_strings_with_domain = [
    f"? x in {domain_a} WHERE {LinkPredicate}(x, '{treats}', '{drug_a}')",
    (
        f"? x in {domain_a}, y in {domain_b} "
        f"WHERE {LinkPredicate}(x, '{interacts}', y) AND {LinkPredicate}(y, '{assoc}', '{drug_a}')"
    ),
    (
        f"? x in {domain_a}, y in {domain_b} "
        f"WHERE NOT {LinkPredicate}(x, '{interacts}', y) "
        f"AND {LinkPredicate}(y, '{assoc}', '{drug_a}')"
    ),
    (
        f"? x in {domain_a}, y in {domain_a} "
        f"WHERE P(x, '{treats}', y) AND P(x, '{treats}', y) AND NOT(P(x, '{assoc}', '{drug_a}'))"
    ),
]
query_string_and_variables_list = [
    (f"{LinkPredicate}(x, '{treats}', '{drug_a}')", [variable_x]),
    (
        f"{LinkPredicate}(x, '{interacts}', y) AND {LinkPredicate}(y, '{assoc}', '{drug_a}')",
        [variable_x, variable_y],
    ),
    (
        f"NOT {LinkPredicate}(x, '{interacts}', y) AND {LinkPredicate}(y, '{assoc}', '{drug_a}')",
        [variable_x, variable_y],
    ),
    (
        f"P(x, '{treats}', y) AND P(x, '{treats}', y) AND NOT(P(x, '{assoc}', '{drug_a}'))",
        [variable_x, variable_y],
    ),
]


@pytest.mark.parametrize("query_string", query_strings_with_domain)
def test_grounding(query_string):
    query = parser.parse_query(query_string)
    pattern_strings = query_string[1:].split("WHERE")[0].split(",")
    variable_domain_mapping = dict(
        tuple(token.strip() for token in pattern_string.split(" in "))
        for pattern_string in pattern_strings
    )

    grounded_query = ground_query(query)

    for variable_label, variable in grounded_query.variables.items():
        assert variable.domain_identifier == variable_domain_mapping[variable_label]


@pytest.mark.parametrize("query_string_and_variables", query_string_and_variables_list)
def test_grounding_given_variables(query_string_and_variables):
    query_string, variables = query_string_and_variables
    query = parser.parse_query(query_string)
    label_to_variable = {variable.label: variable for variable in variables}

    grounded_query = ground_query_given_variables(query=query, variables=variables)

    for grounded_variable_label, grounded_variable in grounded_query.variables.items():
        assert grounded_variable == label_to_variable[grounded_variable_label]
