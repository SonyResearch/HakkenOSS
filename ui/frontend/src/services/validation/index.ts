import { fetchGateway } from '../../utils/apiFetch';

const CONSTANTS = {
  VALIDATION: 'validation',
};

export const validateTriples = async (triples: [string, string, string][]) => {
  const payload = {
    request: {
      facts_list: triples,
      normalize: true,
    },
  };
  const tripleScores = await fetchGateway(
    `${CONSTANTS.VALIDATION}/`,
    'POST',
    payload,
    {},
    {},
  );
  return tripleScores.normalized_scores_list;
};
