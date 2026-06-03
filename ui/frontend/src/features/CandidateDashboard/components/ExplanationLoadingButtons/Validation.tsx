/*Component to trigger explanation requests for the triple requested in validation*/

import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import { SetStateAction } from 'react';
import './index.css';
import ExplanationLoadingControls from './ExplanationLoadingControls';

interface ValidationExplanationLoadingButtonsProps {
  explanationsRequestedKeys: string[];
  selectedTriple: ScoredTriple;
  setPageToShow: React.Dispatch<SetStateAction<'context' | 'explanation'>>;
  currentHypothesisIndexes: number[];
  setCurrentHypothesisIndexes: React.Dispatch<SetStateAction<number[]>>;
}

const ValidationExplanationLoadingButtons = ({
  selectedTriple,
  setPageToShow,
}: ValidationExplanationLoadingButtonsProps) => {
  const { loadExplanations, explanations } = useExplanationContext();
  const strTriple: [string, string, string] = [
    selectedTriple.triple[0].id,
    selectedTriple.triple[1],
    selectedTriple.triple[2].id,
  ];
  const handleLoadExplanation = async () => {
    loadExplanations([strTriple]);
  };
  const key = strTriple.join('-');
  const selectedExplanation = explanations.get(key);

  return (
    <ExplanationLoadingControls
      selectedExplanation={selectedExplanation}
      onRequest={handleLoadExplanation}
      onView={() => setPageToShow('explanation')}
    />
  );
};

export default ValidationExplanationLoadingButtons;
