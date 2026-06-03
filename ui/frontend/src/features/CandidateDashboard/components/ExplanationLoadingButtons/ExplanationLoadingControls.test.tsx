import { describe, vi, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ExplanationLoadingControls from './ExplanationLoadingControls';
import userEvent from '@testing-library/user-event';

describe('ExplanationControls', () => {
  let onRequest: ReturnType<typeof vi.fn>;
  let onView: ReturnType<typeof vi.fn>;

  const mockExplanation = (status: 'loading' | 'ready' | 'error') => ({
    status: status,
    explanation: {
      predictedTriple: { head: 'a', relation: 'treats', tail: 'b' },
      explanations: [
        {
          data: [{ head: 'a', relation: 'treats', tail: 'b' }],
          length: 1,
          score: 9,
        },
      ],
      queryId: 'a-treats-b',
    },
    error: 'Something occured',
    initialTime: 500,
    remainingTime: 0,
  });

  beforeEach(() => {
    onRequest = vi.fn();
    onView = vi.fn();
  });
  it('Should show a button to get an explanation for that triple when not loaded', async () => {
    render(
      <ExplanationLoadingControls
        onView={onView}
        onRequest={onRequest}
        selectedExplanation={undefined}
      />,
    );
    const getExplanationBtn = screen.getByText('Get Explanation Pathways');
    expect(getExplanationBtn).toBeInTheDocument();
    await userEvent.click(getExplanationBtn);
    expect(onRequest).toHaveBeenCalledTimes(1);
  });
  it('Should show an error button when there is an error on the explanation that will request the explanation again', async () => {
    render(
      <ExplanationLoadingControls
        onView={onView}
        onRequest={onRequest}
        selectedExplanation={mockExplanation('error')}
      />,
    );
    const errorBtn = screen.getByText('Try Again');
    expect(errorBtn).toBeInTheDocument();
    await userEvent.click(errorBtn);
    expect(onRequest).toHaveBeenCalledTimes(1);
  });
  it('Should show the loading banner when the explanations are loading', () => {
    render(
      <ExplanationLoadingControls
        onView={onView}
        onRequest={onRequest}
        selectedExplanation={mockExplanation('loading')}
      />,
    );
    const loadingBannerText = screen.getByText('Generating the explanation');
    expect(loadingBannerText).toBeInTheDocument();
  });
  it('Should show a button to view the pathways when the explanation is loaded', async () => {
    render(
      <ExplanationLoadingControls
        onView={onView}
        onRequest={onRequest}
        selectedExplanation={mockExplanation('ready')}
      />,
    );
    const viewExplanationBtn = screen.getByText('View Explanation Pathways');
    expect(viewExplanationBtn).toBeInTheDocument();
    await userEvent.click(viewExplanationBtn);
    expect(onView).toHaveBeenCalledTimes(1);
  });
});
