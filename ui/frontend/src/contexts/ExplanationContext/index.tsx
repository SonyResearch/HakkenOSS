/*Context provider for explanations, gets shortest path and explanations from explainer and sets and updates its status (loading, ready, error) for display,
 aborts the request after estimated time + 30s (estimated time returned from the gateway api, which currently mocks it) */

import { createContext, useContext, useEffect, useState } from 'react';
import {
  ExplanationComplexityResponse,
  ParsedExplanation,
  RawExplanationResult,
} from './types';
import {
  parseExplanationStringtoTriples,
  parsePredictedTripleStringToTriple,
} from '../../services/converters';
import { fetchGateway } from '../../utils/apiFetch';

export type ExplanationState = {
  status: 'loading' | 'ready' | 'error';
  explanation?: ParsedExplanation;
  error?: string;
  initialTime: number;
  remainingTime: number;
};

type ExplanationContext = {
  shortestPathLength: number[];
  getShortestPathLength: (triple: [string, string, string][]) => void;
  explanations: Map<string, ExplanationState>;
  loadExplanations: (triple: [string, string, string][]) => void;
};

const CONSTANTS = {
  EXPLANATION: 'explanation',
  EXPLAIN: 'explain',
};

const ExplanationContext = createContext<ExplanationContext | null>(null);

export const ExplanationProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [shortestPathLength, setShortestPathLength] = useState<number[]>([]);
  const [explanations, setExplanations] = useState<
    Map<string, ExplanationState>
  >(new Map());

  const areThereLoadingExplanations = Array.from(explanations).some(
    (explanation) => explanation[1].status === 'loading',
  );

  useEffect(() => {
    if (areThereLoadingExplanations) {
      const interval = setInterval(() => {
        setExplanations((prevExplanations) => {
          const updatedExplanations = new Map(prevExplanations);
          updatedExplanations.forEach((value, key) => {
            if (value.status === 'loading' && value.remainingTime > 0) {
              updatedExplanations.set(key, {
                ...value,
                remainingTime: Math.max(0, value.remainingTime - 3000),
              });
            }
          });
          return updatedExplanations;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [areThereLoadingExplanations]);

  const getShortestPathLength = async (triples: [string, string, string][]) => {
    const payload = { triples_to_probe: triples };
    try {
      const data: ExplanationComplexityResponse = await fetchGateway(
        `${CONSTANTS.EXPLAIN}/length`,
        'POST',
        payload,
        {},
        {},
      );
      const shortestPathNumbers = Object.values(data.length_dict).map(
        (val) => val,
      );
      setShortestPathLength(shortestPathNumbers);
    } catch (error) {
      console.error('Something went wrong getting complexity lengths', error);
      setShortestPathLength(Array(triples.length).fill('?'));
    }
  };

  const loadExplanations = async (triples: [string, string, string][]) => {
    if (triples.length > 1) {
      const allExplanations = await Promise.all(
        triples.map((triple) => loadSingleExplanation(triple)),
      );
      return allExplanations;
    }
    return loadSingleExplanation(triples[0]);
  };

  const loadSingleExplanation = async (triple: [string, string, string]) => {
    const queryId = triple.join('-');
    const existingExplanation = explanations.get(queryId);
    if (existingExplanation?.status === 'loading') return;
    try {
      const estimatedTime = await fetchGateway(
        `${CONSTANTS.EXPLAIN}/time`,
        'POST',
        triple,
        {},
        {},
      );

      setExplanations((prevExplanations) =>
        new Map(prevExplanations).set(queryId, {
          status: 'loading',
          initialTime: estimatedTime,
          remainingTime: estimatedTime, //change for actual time when we can get that info
        }),
      );
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        estimatedTime + 30000,
      );
      const payload = { triple, estimatedTime };
      const data: RawExplanationResult = await fetchGateway(
        `${CONSTANTS.EXPLAIN}/`,
        'POST',
        payload,
        { signal: controller.signal },
        {},
      );
      clearTimeout(timeoutId);

      const predictedTriple = parsePredictedTripleStringToTriple(
        Object.keys(data.explanations)[0],
      );
      const explanations = Object.values(data.explanations)[0].map(
        (explanation) => ({
          ...explanation,
          data: parseExplanationStringtoTriples(explanation.data),
        }),
      );
      const newExplanation = {
        predictedTriple,
        explanations,
        queryId,
      };

      setExplanations((prevExplanations) =>
        new Map(prevExplanations).set(queryId, {
          status: 'ready',
          explanation: newExplanation,
          initialTime: 0,
          remainingTime: 0,
        }),
      );
    } catch (error) {
      setExplanations((prevExplanations) =>
        new Map(prevExplanations).set(queryId, {
          status: 'error',
          error: (error as Error).message,
          initialTime: 0,
          remainingTime: 0,
        }),
      );
    }
  };
  return (
    <ExplanationContext.Provider
      value={{
        explanations,
        loadExplanations,
        getShortestPathLength,
        shortestPathLength,
      }}
    >
      {children}
    </ExplanationContext.Provider>
  );
};

export const useExplanationContext = (): ExplanationContext => {
  const context = useContext(ExplanationContext);
  if (!context) {
    throw new Error('Explanation context has to be used inside the provider');
  }
  return context;
};
