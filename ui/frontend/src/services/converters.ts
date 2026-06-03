import {
  ParsedCandidate,
  CandidateDTO,
  Triple,
  PredictionType,
} from '../contexts/QueryContext/types';
import { SimpleConcept } from '../static/datasets';
import {
  ConstraintConcept,
  ConstraintRelation,
  ConstraintsFilteringResponse,
} from '../static/datasets/data-types';

export const candidateDTO2Candidate = (
  source: CandidateDTO,
): ParsedCandidate => {
  return {
    variableAssignments: source.var_assignments,
    queryScore: source.query_score,
    conditionsScores: source.condition_scores,
  };
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const snakeToCamel = (obj: any): any => {
  if (Array.isArray(obj)) {
    return obj.map(snakeToCamel);
  } else if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => {
        const camelKey = key.replace(/_([a-z])/g, (_, letter) =>
          letter.toUpperCase(),
        );
        return [camelKey, snakeToCamel(value)];
      }),
    );
  } else {
    return obj;
  }
};

export const parseExplanationStringtoTriples = (string: string): Triple[] => {
  const triplesString = string
    .split('<>')
    .map((s) => s.replace(/[[\]]/g, '').trim());
  const triples: Triple[] = triplesString.map((tripleString) => {
    const match = tripleString.match(
      /^([a-zA-Z0-9]+)-([^-]+)->([a-zA-Z0-9]+)$/,
    );
    if (!match) {
      throw new Error('Invalid format in explanation');
    }
    const [, head, relation, tail] = match;
    return {
      head,
      relation,
      tail,
    };
  });
  return triples;
};

export const parsePredictedTripleStringToTriple = (string: string): Triple => {
  const match = string
    .trim()
    .match(/^([a-z0-9]+)\s*-\s\[([A-Z_]+)\]\s*->\s*([a-z0-9]+)/);
  if (!match) {
    throw new Error('Invalid format in explanation predicted triples');
  }
  const [, head, relation, tail] = match;
  return {
    head,
    relation,
    tail,
  };
};

export const getFieldsFromConstraintResponse = (
  data: ConstraintsFilteringResponse[],
  predictionType: PredictionType,
) => {
  const conceptVariable = predictionType === PredictionType.SUBJECT ? 'Y' : 'X';

  const { possibleRelations, possibleConceptNames } = data.reduce(
    (acc, value) => {
      if (value.type === 'relation') {
        value.values.map((val: ConstraintRelation) =>
          acc.possibleRelations.push(val.label as string),
        );
      } /*else if (value.variable === conceptVariable) {
+        value.values.map((val: ConstraintConcept) => {
+          acc.possibleConcepts.push({
+            node_name: val.label,
+            node_id: val.identifier,
+            node_domain: val.domain_identifier,
+          } as Concept);
+        });
+      }*/ else if (value.variable === conceptVariable) {
        value.values.map((val: ConstraintConcept) => {
          acc.possibleConceptNames.push(val.label);
        });
      }
      return acc;
    },
    {
      possibleRelations: [] as string[],
      possibleConceptNames: [] as string[],
    },
  );

  /*const domainNames = Array.from(
    new Set(possibleConcepts.map((concept: Concept) => concept.node_domain)),
  );
  const possibleDomains: Record<string, Domain> = getDomainsFromDomainNames(domainNames, domains)*/
  return { possibleRelations, possibleConceptNames /*possibleDomains*/ };
};

export const formatTripleForConstraintRequest = (
  subject: SimpleConcept | undefined,
  relation: string | undefined,
  object: SimpleConcept | undefined,
) => {
  const constraintTriple = {
    subject: {
      value: subject && subject.id ? subject.id : 'X',
      is_variable: subject && subject.id ? false : true,
      ...(subject && subject.domain && { domain: subject.domain }),
    },
    relation: {
      value: relation ? relation : 'R',
      is_variable: relation ? false : true,
    },
    object: {
      value: object && object.id ? object.id : 'Y',
      is_variable: object && object.id ? false : true,
      ...(object && object.domain && { domain: object.domain }),
    },
  };
  return constraintTriple;
};
