import './index.css';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import ExplanationRow from '../ExplanationRow';

interface ExplanationSectionProps {
  explanationsRequestedKeys: string[];
  currentHypothesisIndexes: number[];
  notKeys: string[];
  isInValidation: boolean;
}

export const ExplanationSection = ({
  explanationsRequestedKeys,
  currentHypothesisIndexes,
  notKeys,
  isInValidation,
}: ExplanationSectionProps) => {
  const { explanations } = useExplanationContext();
  return (
    <div className="explanation-page">
      <h2>Explanation Pathways</h2>
      {explanationsRequestedKeys && (
        <>
          {explanationsRequestedKeys.map((explanationKey, index) => (
            <ExplanationRow
              key={index}
              explanation={explanations.get(explanationKey)?.explanation}
              collapsed={!currentHypothesisIndexes.includes(index)}
              isNegated={notKeys.includes(explanationKey)}
              isInValidation={isInValidation}
            />
          ))}
        </>
      )}
    </div>
  );
};

export default ExplanationSection;
