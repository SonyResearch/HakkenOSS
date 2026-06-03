import './index.css';
import {
  AddValue,
  SearchedParameters,
} from '../../../../contexts/QueryContext/types';
import { CandidateResultType } from '../../types';
import { getTriplesFromPredictedHypotheses } from '../utils';
import { useEffect, useState } from 'react';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import { getHypothesesKeysFromIndexes } from '../ExplanationLoadingButtons/Query';
import QueryContextualizationSection from '../ContextualizationSection/Query';
import { CandidatePageLayout } from '.';

interface CandidateProps {
  selectedCandidate: CandidateResultType;
  searchedParameters: SearchedParameters;
}

export const QueryCandidatePage = ({
  selectedCandidate,
  searchedParameters,
}: CandidateProps) => {
  const [notKeys, setNotKeys] = useState<string[]>([]);
  const { explanations, getShortestPathLength } = useExplanationContext();
  const explanationsRequestedKeys = [...explanations.keys()].filter((key) =>
    key.includes(selectedCandidate.variableAssignments.X),
  );
  const [pageToShow, setPageToShow] = useState<'context' | 'explanation'>(
    'context',
  );

  const getNotKeys = () => {
    const hypotheses = Object.values(searchedParameters.hypotheses);

    const matchingNotIndexes = hypotheses
      .map((h, i) => (h.addValue === AddValue.AND_NOT ? i : null))
      .filter((i): i is number => i !== null);

    const matchingNotKeys = getHypothesesKeysFromIndexes(
      searchedParameters.hypotheses,
      matchingNotIndexes,
      selectedCandidate.variableAssignments.X,
    ).map((keys) => keys.join('-'));
    setNotKeys(matchingNotKeys);
  };

  useEffect(() => {
    getNotKeys();
    getShortestPathLength(
      getTriplesFromPredictedHypotheses(
        Object.values(searchedParameters.hypotheses),
        selectedCandidate.variableAssignments.X,
      ),
    );
    setPageToShow('context');
  }, [selectedCandidate]);

  return (
    <CandidatePageLayout
      explanationsRequestedKeys={explanationsRequestedKeys}
      notKeys={notKeys}
      isInValidation={false}
      pageToShow={pageToShow}
      setPageToShow={setPageToShow}
      renderContext={(props) => (
        <QueryContextualizationSection
          {...props}
          searchedParameters={searchedParameters}
          explanationsRequestedKeys={explanationsRequestedKeys}
          selectedCandidate={selectedCandidate}
          notKeys={notKeys}
        />
      )}
    />
  );
};
