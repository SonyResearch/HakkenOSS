import { render, screen } from '@testing-library/react';
import DashboardLayout from '.';
import { describe, it } from 'vitest';
import {
  AddValue,
  Condition,
  ConditionType,
  SearchedParameters,
} from '../../../../contexts/QueryContext/types';

describe('DashboardLayout', () => {
  const mockCandidate = (name: string, score: number) => ({
    variableAssignments: { X: name },
    queryScore: score,
    conditionsScores: {},
  });

  const mockCondition = (id: string) => ({
    condition: {
      head: {
        label: id,
        domain: 'disease',
        id: '1234',
        isVariable: true,
      },
      relation: 'ASSOCIATE',
      tail: {
        label: 'x',
        domain: 'protein',
        id: '1234',
        isVariable: false,
      },
    },
    conditionType: ConditionType.hypotheses,
    addValue: AddValue.AND,
  });

  const mockConditions: Record<number, Condition> = {
    0: mockCondition('123'),
    1: mockCondition('234'),
    2: mockCondition('345'),
  };

  const mockCandidates = [
    mockCandidate('Ibuprofen', 9.123),
    mockCandidate('Parecetamol', 8.321),
    mockCandidate('Aspirin', 7),
  ];

  const mockParameters: SearchedParameters = {
    query: '1234',
    hypotheses: mockConditions,
    constraints: {},
    candidatesNumber: 8,
  };

  it('Should display the number of candidates passed in the props and the name and score of the first candidate on the context page', () => {
    render(
      <DashboardLayout
        searchedParameters={mockParameters}
        candidatesResult={mockCandidates}
      />,
    );
    const candidates = screen.getAllByTestId('candidate-row');
    expect(candidates).toHaveLength(3);
    const candidate1Score = screen.getAllByText('9.123');
    expect(candidate1Score).toHaveLength(1);
  });
  it('Should display an empty result message in case no candidates are found', () => {
    render(
      <DashboardLayout
        searchedParameters={mockParameters}
        candidatesResult={[]}
      />,
    );
    expect(
      screen.getByText('We could not find any result'),
    ).toBeInTheDocument();
  });
});
