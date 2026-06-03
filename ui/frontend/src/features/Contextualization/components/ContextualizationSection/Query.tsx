import React, { SetStateAction, useEffect } from 'react';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import {
  ConditionType,
  SearchedParameters,
} from '../../../../contexts/QueryContext/types';
import { QueryVisualization } from '../../../Visualizations/QueryVisualization';
import ScoreMarker from '../../../../shared/components/ScoreMarker';
import QueryExplanationLoadingButtons from '../../../CandidateDashboard/components/ExplanationLoadingButtons/Query';
import {
  CandidateResultType,
  Filters,
  Sorting,
} from '../../../CandidateDashboard/types';
import ReferencesTable from '../ReferencesTable';
import { useContextualizationResults } from '../../../../hooks/useContextualizationResults';
import { getTriplesFromPredictedHypotheses } from '../../../CandidateDashboard/components/utils';

interface QueryContextualizationSectionProps {
  searchedParameters: SearchedParameters;
  selectedCandidate: CandidateResultType;
  explanationsRequestedKeys: string[];
  setPageToShow: React.Dispatch<SetStateAction<'context' | 'explanation'>>;
  currentHypothesisIndexes: number[];
  setCurrentHypothesisIndexes: React.Dispatch<SetStateAction<number[]>>;
  filters: Filters;
  sorting: Sorting;
  setFilters: React.Dispatch<SetStateAction<Filters>>;
  setSorting: React.Dispatch<SetStateAction<Sorting>>;
  notKeys: string[];
}

const QueryContextualizationSection = ({
  searchedParameters,
  selectedCandidate,
  explanationsRequestedKeys,
  setPageToShow,
  currentHypothesisIndexes,
  setCurrentHypothesisIndexes,
  filters,
  sorting,
  setFilters,
  setSorting,
  notKeys,
}: QueryContextualizationSectionProps) => {
  const conditionsLengths = {
    [ConditionType.hypotheses]: Object.values(searchedParameters.hypotheses)
      .length,
    [ConditionType.constraints]: Object.values(searchedParameters.constraints)
      .length,
  };
  const { shortestPathLength } = useExplanationContext();
  const { loading, contextualization, fetchContextualization, error } =
    useContextualizationResults();

  useEffect(() => {
    fetchContextualization(
      getTriplesFromPredictedHypotheses(
        Object.values(searchedParameters.hypotheses),
        selectedCandidate.variableAssignments.X,
      ),
    );
  }, [selectedCandidate]);

  return (
    <div className="candidate-page">
      <div>
        <div className="main-information">
          <div className="candidate-name-and-score">
            <div>
              <h2 data-testid="candidate-name">{selectedCandidate?.name}</h2>
              <p>{selectedCandidate?.domain?.replace(/_/g, ' ')}</p>
            </div>
            <ScoreMarker score={selectedCandidate.queryScore} size="big" />
          </div>
          <div>
            <QueryExplanationLoadingButtons
              explanationsRequestedKeys={explanationsRequestedKeys}
              searchedParameters={searchedParameters}
              selectedCandidate={selectedCandidate}
              conditionsLengths={conditionsLengths}
              setPageToShow={setPageToShow}
              currentHypothesisIndexes={currentHypothesisIndexes}
              setCurrentHypothesisIndexes={setCurrentHypothesisIndexes}
              notKeys={notKeys}
            />
          </div>
        </div>
        <QueryVisualization
          page="results"
          selectedCandidate={selectedCandidate}
          hypotheses={searchedParameters.hypotheses}
          constraints={searchedParameters.constraints}
          query={searchedParameters.query}
          conditionsLengths={conditionsLengths}
          selectedCandidateName={selectedCandidate?.name}
          setCurrentHypothesisIndexes={setCurrentHypothesisIndexes}
          shortestPathLength={shortestPathLength}
        />
      </div>
      <ReferencesTable
        contextualization={contextualization}
        error={error}
        loading={loading}
        filters={filters}
        sorting={sorting}
        setFilters={setFilters}
        setSorting={setSorting}
      />
    </div>
  );
};

export default QueryContextualizationSection;
