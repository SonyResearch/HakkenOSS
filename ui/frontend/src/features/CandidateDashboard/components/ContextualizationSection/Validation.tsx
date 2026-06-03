/*Contextualization section for the validation results*/

import { SetStateAction, useEffect } from 'react';
import ScoreMarker from '../../../../shared/components/ScoreMarker';
import ValidationExplanationLoadingButtons from '../ExplanationLoadingButtons/Validation';
import ReferencesTable from '../../../Contextualization/components/ReferencesTable';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import { useContextualizationResults } from '../../../../hooks/useContextualizationResults';
import { makeTripleStringArr } from '../utils';
import { Filters, Sorting } from '../../types';

interface ValidationContextualizationSectionProps {
  selectedTriple: ScoredTriple;
  explanationsRequestedKeys: string[];
  setPageToShow: React.Dispatch<SetStateAction<'context' | 'explanation'>>;
  currentHypothesisIndexes: number[];
  setCurrentHypothesisIndexes: React.Dispatch<SetStateAction<number[]>>;
  filters: Filters;
  sorting: Sorting;
  setFilters: React.Dispatch<SetStateAction<Filters>>;
  setSorting: React.Dispatch<SetStateAction<Sorting>>;
}

const ValidationContextualizationSection = ({
  selectedTriple,
  explanationsRequestedKeys,
  setPageToShow,
  currentHypothesisIndexes,
  setCurrentHypothesisIndexes,
  filters,
  sorting,
  setFilters,
  setSorting,
}: ValidationContextualizationSectionProps) => {
  const tripleStr = makeTripleStringArr(selectedTriple.triple, 'name');
  const tripleStrId = makeTripleStringArr(selectedTriple.triple, 'id');

  const { loading, contextualization, fetchContextualization, error } =
    useContextualizationResults();

  useEffect(() => {
    fetchContextualization([tripleStrId]);
  }, [tripleStrId]);

  return (
    <div className="candidate-page validation">
      <div>
        <div className="main-information">
          <div className="candidate-name-and-score">
            <div>
              <h2>{tripleStr.join('-')}</h2>
            </div>
            <ScoreMarker score={selectedTriple.score ?? 0} size="big" />
          </div>
          <div>
            <ValidationExplanationLoadingButtons
              explanationsRequestedKeys={explanationsRequestedKeys}
              setPageToShow={setPageToShow}
              currentHypothesisIndexes={currentHypothesisIndexes}
              setCurrentHypothesisIndexes={setCurrentHypothesisIndexes}
              selectedTriple={selectedTriple}
            />
          </div>
        </div>
      </div>
      <ReferencesTable
        contextualization={contextualization}
        error={error}
        loading={loading}
        sorting={sorting}
        filters={filters}
        setFilters={setFilters}
        setSorting={setSorting}
      />
    </div>
  );
};

export default ValidationContextualizationSection;
