import { ScoredTriple } from '../../pages/ValidationPage';
import { filterOptionsOfRelations } from '../QueryForm/components/MainForm/utils';
import { PredictionType } from '../../contexts/QueryContext/types';
import { Concept } from '../../static/datasets';
import { SelectedOptions } from './components/ValidationForm';

/*Get all potential relations between two entites*/
export const getPossibleTriplesByPair = async (
  entity1: Concept & { domain: string },
  entity2: Concept & { domain: string },
) => {
  const possibleRelations = await filterOptionsOfRelations(
    entity1.domain,
    entity2.domain,
    PredictionType.SUBJECT,
  );
  const newTriples: ScoredTriple[] = Object.values(possibleRelations).map(
    (relation) => {
      return {
        triple: [entity1, relation, entity2],
        score: undefined,
      };
    },
  );
  return newTriples;
};

export const isEmptySelection = (options: {
  subjectDomain: string;
  objectDomain: string;
  relation: string;
  subjectConcept: Concept;
  objectConcept: Concept;
}) => {
  return (
    !options.subjectDomain ||
    !options.objectDomain ||
    !options.relation ||
    !options.subjectConcept?.id ||
    !options.objectConcept?.id
  );
};

export const mapTriplesToValidationFormat = (
  triples: ScoredTriple[],
): [string, string, string][] =>
  triples.map((t) => [t.triple[0].id, t.triple[1], t.triple[2].id]);

export const buildTriplesFromSelection = async (
  opts: SelectedOptions,
): Promise<ScoredTriple[]> => {
  if (opts.relation === 'ANY') {
    return getPossibleTriplesByPair(
      { ...opts.subjectConcept, domain: opts.subjectDomain },
      { ...opts.objectConcept, domain: opts.objectDomain },
    );
  }

  return [
    {
      triple: [opts.subjectConcept, opts.relation, opts.objectConcept],
      score: undefined,
    },
  ];
};

export const normalizeScore = (rawScore: number | undefined) => {
  if (rawScore === undefined) return '?';

  const prefix = rawScore.toString().slice(0, 5);

  if (prefix === '0.999') return 1.0;
  if (prefix === '0.000') return 0.0;

  return rawScore;
};
