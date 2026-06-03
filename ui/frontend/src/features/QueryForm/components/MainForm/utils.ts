import { appConfig } from '../../../../config';
import {
  AddValue,
  Condition,
  PredictionType,
} from '../../../../contexts/QueryContext/types';
import { Domain, Concepts } from '../../../../static/datasets/data-types';
import { Concept, SimpleConcept } from '../../../../static/datasets';
import { QueryFormType } from '../../../../contexts/QueryFormContext/types';
import {
  formatTripleForConstraintRequest,
  getFieldsFromConstraintResponse,
} from '../../../../services/converters';
import { fetchGateway } from '../../../../utils/apiFetch';

const CONSTANTS = {
  RELATIONS: 'filter-relations',
  DOMAINS: 'filter-domains',
  CONCEPTS: 'filter-concepts',
  TRIPLES: 'check-triples',
  CONSTRAINT: 'filter_constraint',
  DATA: 'data',
  QUERY: 'query',
};

export const isNestedObjectEqual = (a: Domain, b: Domain): boolean => {
  return (
    a.node_domain_id === b.node_domain_id && a.node_domain === b.node_domain
  );
};

export const getOptionsFromConcepts = (concepts: Concept[]) => {
  return [...new Set(concepts.map((concept) => concept.name))];
};

export const limitConceptsUnder20000 = (concepts: Concepts, limit = 20000) => {
  const keys = Object.keys(concepts);
  if (keys.length <= limit) return concepts;
  else {
    const shuffledKeys = keys.sort(() => Math.random() - 0.5);
    const limitedKeys = shuffledKeys.slice(0, limit);

    const limitedData: Concepts = {};
    for (const key of limitedKeys) {
      limitedData[key] = concepts[key];
    }
    return limitedData;
  }
};

export const filterOptionsOfRelations = async (
  variableDomain: string,
  conceptDomain: string,
  predictionType: string,
) => {
  const [subjectDomain, objectDomain] =
    predictionType === PredictionType.SUBJECT
      ? [variableDomain, conceptDomain]
      : [conceptDomain, variableDomain];
  const query = {
    subject: subjectDomain || null,
    object: objectDomain || null,
  };
  const data = await fetchGateway(
    `${CONSTANTS.DATA}/getedgetypes`,
    'GET',
    {},
    {},
    query,
  );
  const filteredRelations: string[] = data.edge_types;
  return filteredRelations;
};

export const filterOptionsOfDomains = async (
  currentDomain: string,
  selectedRelation: string,
  isSubject: boolean,
) => {
  const relation = selectedRelation === 'ANY' ? null : selectedRelation;
  const getAllDomains = !currentDomain && !relation ? true : false;
  const path = getAllDomains ? '/getuniquedomains' : '/getnodedomains';
  const [subjectDomain, objectDomain] = isSubject
    ? [null, currentDomain]
    : [currentDomain, null];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const query: any = getAllDomains
    ? {}
    : {
        subject: subjectDomain,
        object: objectDomain,
        edge: relation || null,
      };
  const data = await fetchGateway(
    `${CONSTANTS.DATA}${path}`,
    'GET',
    {},
    {},
    query,
  );
  const filteredDomains: string[] = getAllDomains
    ? data.domain_names
    : data.node_domains;
  return filteredDomains;
};

export const updateConcepts = async (domain: string, limit?: number) => {
  const query = {
    domain: domain,
    max_results: limit ?? appConfig.conceptLimit,
  };
  const data = await fetchGateway(
    `${CONSTANTS.DATA}/getnodesfromdomain`,
    'GET',
    {},
    {},
    query,
  );
  const updatedConcepts: Concept[] = data.nodes;
  return updatedConcepts;
};

export const searchConcepts = async (domain: string, query: string) => {
  if (!domain) {
    throw new Error(
      'You need to select a domain in order to search for concepts',
    );
  }
  const queryParams = {
    domain: domain,
    node: query,
    max_results: appConfig.conceptLimit,
  };
  const data = await fetchGateway(
    `${CONSTANTS.DATA}/getnodesfromdomain`,
    'GET',
    {},
    {},
    queryParams,
  );
  const updatedConcepts: Concept[] = data.nodes;
  return updatedConcepts;
};

export const getNodeNames = async (nodeIds: string[]) => {
  const payload = {
    id_list: nodeIds,
  };
  const data = await fetchGateway(
    `${CONSTANTS.DATA}/getname`,
    'POST',
    payload,
    {},
    {},
  );
  const nodeNames: Record<string, string> = data.id_name_mapping;
  return nodeNames;
};

export const createNewCondition = async (
  selectedVariableDomain: string,
  selectedRelation: string,
  selectedConceptDomain: string,
  conceptName: string,
  predictionType: string,
) => {
  if (
    [
      selectedVariableDomain,
      selectedRelation,
      selectedConceptDomain,
      conceptName,
      predictionType,
    ].some((value) => !value)
  ) {
    throw new Error('All fields must be selected to create a condition');
  }
  const condition = checkCondition(
    selectedVariableDomain,
    selectedRelation,
    selectedConceptDomain,
    conceptName,
    predictionType,
  );
  return condition;
};

export const checkCondition = async (
  variableDomain: string,
  relation: string,
  conceptDomain: string,
  conceptName: string,
  predictionType: string,
) => {
  const [subjectDomain, objectDomain] =
    predictionType === PredictionType.SUBJECT
      ? [variableDomain, conceptDomain]
      : [conceptDomain, variableDomain];
  const concepts: Concept[] = await updateConcepts(conceptDomain, 500000);
  const matchingConcept = concepts.find(
    (concept) => concept.name === conceptName,
  ) as Concept;
  const possibleRelations: string[] = await filterOptionsOfRelations(
    variableDomain,
    conceptDomain,
    predictionType,
  );
  const matchingRelation = possibleRelations.includes(relation);
  if (!matchingConcept) {
    throw new Error(
      `Concept '${conceptName}' not found in domain '${conceptDomain}'.`,
    );
  }

  if (!matchingRelation) {
    throw new Error(
      `Relation '${relation}' is not valid between '${subjectDomain}' and '${objectDomain}'.`,
    );
  }
  const condition = {
    tail: {
      domain: conceptDomain,
      id: matchingConcept.id,
      isVariable: false,
      label: matchingConcept.name,
    },
    relation: relation,
    head: {
      domain: variableDomain,
      id: variableDomain,
      isVariable: true,
      label: 'X',
    },
  };
  return condition;
};

export const getConditionText = (
  condition: Condition,
  candidate?: string | undefined,
) => {
  let [headLabel, tailLabel] = [
    condition.condition.head.label,
    condition.condition.tail.label,
  ];
  if (candidate) {
    if (condition.condition.head.isVariable) {
      headLabel = candidate;
    } else {
      tailLabel = candidate;
    }
  }
  const conditionText = `${headLabel} ∈ ${condition.condition.head.domain.replace(/_/g, ' ')}, ${condition.condition.relation.replace(/_/g, ' ')}, ${tailLabel} ∈ ${condition.condition.tail.domain.replace(/_/g, ' ')}`;

  return conditionText;
};

export const getAddValue = (listing: number, addValue: AddValue) => {
  if (listing !== 0) return addValue;
  if (addValue === AddValue.AND_NOT) {
    return AddValue.AND_NOT.replace(/AND/, '').trim();
  }
  return '';
};

export const getTripleFromCondition = (condition: Condition) => {
  const [variable, concept, relation, isSubjectPrediction] = condition.condition
    .head.isVariable
    ? [
        condition.condition.head,
        condition.condition.tail,
        condition.condition.relation,
        true,
      ]
    : [
        condition.condition.tail,
        condition.condition.head,
        condition.condition.relation,
        false,
      ];
  return { variable, concept, relation, isSubjectPrediction };
};

export const isThereAnyVariable = (
  conditionTypes: Record<number, Condition>[],
) => {
  const conditionsCount = conditionTypes.reduce(
    (count, conditionType) => count + Object.values(conditionType).length,
    0,
  );
  for (const conditionType of conditionTypes) {
    const firstCondition = Object.values(conditionType)[0];
    if (firstCondition && conditionsCount > 1) {
      const { variable } = getTripleFromCondition(
        Object.values(conditionType)[0],
      );
      if (variable.domain !== '') return true;
    }
  }
  return false;
};

export const filterOptionsOfConstraints = async (
  form: QueryFormType,
  variableDomain: string,
  concepts: Concept[],
) => {
  const matchingConcept = concepts.find(
    (concept) => concept.name === form.tailValue[0],
  ) as Concept;
  const variableConcept: SimpleConcept = {
    domain: variableDomain,
    id: '',
  };
  const definedConcept: SimpleConcept = {
    domain: form.selectedConceptDomain,
    id: matchingConcept ? matchingConcept.id : '',
  };
  const relation: string = matchingConcept ? '' : form.selectedRelation; //to leave the relation empty if all other inputs are filled.

  const [subject, object] =
    form.predictionType === PredictionType.SUBJECT
      ? [variableConcept, definedConcept]
      : [definedConcept, variableConcept];

  const payload = formatTripleForConstraintRequest(subject, relation, object);

  const data = await fetchGateway(
    `${CONSTANTS.QUERY}/${CONSTANTS.CONSTRAINT}`,
    'POST',
    payload,
    {},
    {},
  );
  const { possibleRelations, possibleConceptNames } =
    getFieldsFromConstraintResponse(
      data,
      form.predictionType as PredictionType,
    );
  return { possibleRelations, possibleConceptNames };
};
