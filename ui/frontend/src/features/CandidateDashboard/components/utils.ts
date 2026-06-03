import { Condition, Triple } from '../../../contexts/QueryContext/types';
import { Concept, Concepts } from '../../../static/datasets';
import { ConceptNode } from '../../../static/datasets/data-types';
import { getTripleFromCondition } from '../../QueryForm/components/MainForm/utils';

export const getTriplesFromPredictedHypotheses = (
  hypotheses: Condition[],
  predictedEntityId: string,
) => {
  const triples: [string, string, string][] = [];
  hypotheses.forEach((hypothesis) => {
    const triple = getTripleFromPredictedHypothesis(
      hypothesis,
      predictedEntityId,
    );
    triples.push(triple);
  });
  return triples;
};

export const getTripleFromPredictedHypothesis = (
  hypothesis: Condition,
  predictedEntityId: string,
): [string, string, string] => {
  const { relation, concept, isSubjectPrediction } =
    getTripleFromCondition(hypothesis);
  const triple: Triple = {
    head: isSubjectPrediction
      ? predictedEntityId
      : concept.id.replace('id', ''),
    relation: relation,
    tail: isSubjectPrediction
      ? concept.id.replace('id', '')
      : predictedEntityId,
  };
  return [triple.head, triple.relation, triple.tail];
};

export const findConceptById = (
  concepts: Concepts | ConceptNode[],
  id: string,
) => {
  const concept = Object.values(concepts).find(
    (concept) => concept.node_id === `id${id}` || concept.node_id === id,
  );
  return concept;
};

export const makeTripleStringArr = (
  triple: [Concept, string, Concept],
  key: 'name' | 'id',
) => [triple[0][key], triple[1], triple[2][key]] as [string, string, string];
