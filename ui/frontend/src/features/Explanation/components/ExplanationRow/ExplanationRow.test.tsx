/* eslint-disable @typescript-eslint/no-explicit-any */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ExplanationRow from '.';

vi.mock('../../../Visualizations/ExplainabilityGraph', () => ({
  ExplainabilityGraph: ({ selectedExplanation }: any) => (
    <div data-testid="mock-graph">
      Mock Graph - {selectedExplanation.queryId}
    </div>
  ),
}));

const mockExplanation = {
  predictedTriple: {
    head: 'T',
    relation: 'TREATS',
    tail: 'B',
  },
  queryId: 'A-TREATS-B',
  explanations: [{ data: ['abc', 'treats', 'bcd'], length: 3, score: 0.99 }],
};

describe('ExplanationRow', () => {
  it('returns null when explanation is undefined', () => {
    const { container } = render(
      <ExplanationRow
        explanation={undefined}
        collapsed={false}
        isNegated={false}
        isInValidation={false}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders queryId entities', () => {
    render(
      <ExplanationRow
        explanation={mockExplanation as any}
        collapsed={false}
        isNegated={false}
        isInValidation={false}
      />,
    );

    expect(screen.getByText(/-TREATS-B/)).toBeInTheDocument();
  });

  it('toggles collapse on button click', () => {
    render(
      <ExplanationRow
        //typescript-disable
        explanation={mockExplanation as any}
        collapsed={true}
        isNegated={false}
        isInValidation={false}
      />,
    );

    const button = screen.getByRole('button');
    expect(button).toHaveTextContent('Show');

    fireEvent.click(button);
    expect(button).toHaveTextContent('Hide');
  });

  it('shows fallback message when explanations are empty', () => {
    render(
      <ExplanationRow
        explanation={{ ...mockExplanation, explanations: [] } as any}
        collapsed={false}
        isNegated={false}
        isInValidation={false}
      />,
    );

    expect(
      screen.getByText(/Sorry, our system could not find an explanation/i),
    ).toBeInTheDocument();
  });

  it('renders graph when explanations exist', () => {
    render(
      <ExplanationRow
        explanation={mockExplanation as any}
        collapsed={false}
        isNegated={false}
        isInValidation={false}
      />,
    );

    expect(screen.getByTestId('mock-graph')).toBeInTheDocument();
  });

  it('shows tooltip when negated and hovered', () => {
    render(
      <ExplanationRow
        explanation={mockExplanation as any}
        collapsed={false}
        isNegated={true}
        isInValidation={false}
      />,
    );

    const icon = screen.getByText('!');
    fireEvent.mouseEnter(icon);

    expect(
      screen.getByText(/cannot process negated relationships/i),
    ).toBeInTheDocument();

    fireEvent.mouseLeave(icon);
    expect(
      screen.queryByText(/cannot process negated relationships/i),
    ).not.toBeInTheDocument();
  });
});
