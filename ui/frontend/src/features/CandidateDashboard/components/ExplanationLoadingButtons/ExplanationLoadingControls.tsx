/*Visual loading state for explanation where we show buttons or a loading bar depending on the explanation status (loading, ready or error)*/

import { Button } from '@mui/material';
import { LoadingBanner } from '../../../../shared/components/LoadingBanner';
import { ExplanationState } from '../../../../contexts/ExplanationContext';

interface ExplanationLoadingControlsProps {
  selectedExplanation: ExplanationState | undefined;
  onRequest: () => void;
  onView: () => void;
}

const loadingTexts = [
  'Processing triples...',
  'Sorting relevant nodes...',
  'Selecting best explanations...',
];

const ExplanationLoadingControls = ({
  selectedExplanation,
  onRequest,
  onView,
}: ExplanationLoadingControlsProps) => {
  if (!selectedExplanation) {
    return (
      <Button
        data-testid="get-explanation-button"
        variant="contained"
        onClick={onRequest}
      >
        Get Explanation Pathways
      </Button>
    );
  }

  if (selectedExplanation.status === 'loading') {
    return (
      <LoadingBanner
        initialTime={selectedExplanation.initialTime}
        remainingTime={selectedExplanation.remainingTime}
        texts={loadingTexts}
      />
    );
  }

  if (selectedExplanation.status === 'error') {
    return (
      <div className="explanation-error">
        <p>{selectedExplanation.error}</p>
        <Button
          variant="contained"
          onClick={onRequest}
          sx={{ backgroundColor: 'firebrick', color: 'white' }}
          data-testid="error-explanation-button"
        >
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <Button
      data-testid="view-explanation-button"
      variant="contained"
      onClick={onView}
    >
      View Explanation Pathways
    </Button>
  );
};

export default ExplanationLoadingControls;
