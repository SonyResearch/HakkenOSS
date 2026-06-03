import { useState } from 'react';
import { CandidatePageLayout } from '.';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import ValidationContextualizationSection from '../ContextualizationSection/Validation';

interface ValidationCandidateProps {
  selectedTriple: ScoredTriple;
}

export const ValidationCandidatePage = ({
  selectedTriple,
}: ValidationCandidateProps) => {
  const { explanations } = useExplanationContext();
  const [pageToShow, setPageToShow] = useState<'context' | 'explanation'>(
    'context',
  );

  const tripleStrId = [
    selectedTriple.triple[0].id,
    selectedTriple.triple[1],
    selectedTriple.triple[2].id,
  ];

  const explanationsRequestedKeys = [...explanations.keys()].filter((key) =>
    key.includes(tripleStrId.join('-')),
  );

  return (
    <CandidatePageLayout
      explanationsRequestedKeys={explanationsRequestedKeys}
      notKeys={[]}
      isInValidation
      pageToShow={pageToShow}
      setPageToShow={setPageToShow}
      renderContext={(props) => (
        <ValidationContextualizationSection
          {...props}
          explanationsRequestedKeys={explanationsRequestedKeys}
          selectedTriple={selectedTriple}
        />
      )}
    />
  );
};
