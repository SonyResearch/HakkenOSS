import { useState } from 'react';
import { ContextualizationResult } from '../../features/CandidateDashboard/types';
import { TripleContext } from '../../contexts/QueryContext/types';
import { fetchGateway } from '../../utils/apiFetch';

const CONSTANTS = {
  CONTEXTUALIZATION: 'contextualization',
  CONTEXT: 'context',
  CONTEXTUALIZE: 'contextualize',
};

const contextualizationCache = new Map<string, ContextualizationResult>();

const getKeyFromTriples = (triples: [string, string, string][]) => {
  return triples.map((triple) => triple.join('-')).join('--');
};

export const useContextualizationResults = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [contextualization, setContextualization] =
    useState<ContextualizationResult>();

  const fetchContextualization = async (
    triples: [string, string, string][],
  ) => {
    setLoading(true);
    setError('');
    const cacheKey = getKeyFromTriples(triples); //cache result only on react state in case user switches between candidates frequently
    const formattedTriples: TripleContext[] = triples.map((triple) => {
      return {
        subject: triple[0],
        relation: triple[1],
        object: triple[2],
      };
    });

    try {
      if (contextualizationCache.has(cacheKey)) {
        setContextualization(contextualizationCache.get(cacheKey));
      } else {
        const payload = {
          triples: formattedTriples,
          max_num_references: 20,
          return_type: 'summary',
        };
        const contextualization: ContextualizationResult = await fetchGateway(
          `${CONSTANTS.CONTEXT}/${CONSTANTS.CONTEXTUALIZE}`,
          'POST',
          payload,
          {},
          {},
        );

        if (
          !contextualization ||
          !Array.isArray(contextualization.references) ||
          contextualization.references.length === 0
        ) {
          console.warn(
            `Empty results contextualizing this triple: ${cacheKey}`,
          );
          throw new Error('Empty contextualization result');
        }
        contextualizationCache.set(cacheKey, contextualization);
        setContextualization(contextualization);
      }
    } catch (error) {
      setError(
        (error as Error).message ??
          'Something went wrong getting contextualization',
      );
    } finally {
      setLoading(false);
    }
  };

  return { contextualization, loading, error, fetchContextualization };
};
