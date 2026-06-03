/*Component to trigger explanation requests, if query has more than one condition, the component will display a dropdown to chose which triple to request explanation for,
the user can also choose all of them */

import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import { getTripleFromPredictedHypothesis } from '../utils';
import React, { SetStateAction, useEffect, useState } from 'react';
import './index.css';
import {
  AddValue,
  Condition,
  ConditionType,
  SearchedParameters,
} from '../../../../contexts/QueryContext/types';
import { getConditionText } from '../../../QueryForm/components/MainForm/utils';
import { CandidateResultType } from '../../types';
import { useConfirm } from '../../../../hooks/useConfirm';
import ExplanationLoadingControls from './ExplanationLoadingControls';

interface QueryExplanationLoadingButtonsProps {
  explanationsRequestedKeys: string[];
  conditionsLengths: Record<ConditionType, number>;
  searchedParameters: SearchedParameters;
  selectedCandidate: CandidateResultType;
  setPageToShow: React.Dispatch<SetStateAction<'context' | 'explanation'>>;
  currentHypothesisIndexes: number[];
  setCurrentHypothesisIndexes: React.Dispatch<SetStateAction<number[]>>;
  notKeys: string[];
}

export const getHypothesesKeysFromIndexes = (
  hypotheses: Record<number, Condition>,
  indexes: number[],
  candidateOcid: string,
) => {
  const keys = indexes.map((index) =>
    getTripleFromPredictedHypothesis(
      Object.values(hypotheses)[index],
      candidateOcid,
    ),
  );
  return keys;
};

const QueryExplanationLoadingButtons = ({
  conditionsLengths,
  searchedParameters,
  selectedCandidate,
  setPageToShow,
  currentHypothesisIndexes,
  setCurrentHypothesisIndexes,
  notKeys,
}: QueryExplanationLoadingButtonsProps) => {
  const { confirm, ConfirmDialog } = useConfirm();
  const { loadExplanations } = useExplanationContext();
  const [currentHypothesisKeys, setCurrentHypothesisKeys] = useState<
    [string, string, string][]
  >(
    getHypothesesKeysFromIndexes(
      searchedParameters.hypotheses,
      currentHypothesisIndexes,
      selectedCandidate.variableAssignments.X,
    ),
  );
  const { explanations } = useExplanationContext();
  const selectedExplanation = explanations.get(
    currentHypothesisKeys[0].join('-'),
  );

  useEffect(() => {
    setCurrentHypothesisKeys(
      getHypothesesKeysFromIndexes(
        searchedParameters.hypotheses,
        currentHypothesisIndexes,
        selectedCandidate.variableAssignments.X,
      ),
    );
  }, [currentHypothesisIndexes, selectedCandidate]);

  const handleLoadExplanation = async () => {
    let foundOnlyOneNOT = false;
    if (notKeys.length) {
      for (const notKey of notKeys) {
        if (
          notKey &&
          currentHypothesisKeys.some((key) => key.join('-') === notKey)
        ) {
          if (currentHypothesisKeys.length === 1) {
            foundOnlyOneNOT = true;
            const dialogMessage = [
              'The selected hypothesis involves a negated condition (“NOT”). At this time, our explanation engine cannot process negated relationships.',
              'Would you like to generate an explanation for the equivalent positive condition instead?.',
            ];
            const userConfirmed = await confirm(dialogMessage);
            if (userConfirmed) {
              loadExplanations(currentHypothesisKeys);
            }
            return;
          }
        }
      }
    }
    if (!foundOnlyOneNOT) {
      loadExplanations(currentHypothesisKeys);
    }
  };

  return (
    <div>
      {ConfirmDialog}
      {conditionsLengths[ConditionType.hypotheses] > 1 && (
        <select
          value={
            currentHypothesisIndexes.length > 1
              ? 'ALL'
              : currentHypothesisIndexes[0]
          }
          className="condition-select"
          onChange={(e) =>
            setCurrentHypothesisIndexes(
              e.target.value === 'ALL'
                ? [
                    ...Array(
                      Object.values(searchedParameters.hypotheses).length,
                    ).keys(),
                  ]
                : [Number(e.target.value)],
            )
          }
        >
          {Object.values(searchedParameters?.hypotheses).map(
            (hypothesis, index) => {
              return (
                <option key={index} value={index}>
                  {hypothesis.addValue === AddValue.AND_NOT
                    ? `NOT (${getConditionText(hypothesis, selectedCandidate?.name)})`
                    : getConditionText(hypothesis, selectedCandidate?.name)}
                </option>
              );
            },
          )}
          <option value="ALL">All explanations</option>
        </select>
      )}
      <ExplanationLoadingControls
        selectedExplanation={selectedExplanation}
        onRequest={handleLoadExplanation}
        onView={() => setPageToShow('explanation')}
      />
    </div>
  );
};

export default QueryExplanationLoadingButtons;
