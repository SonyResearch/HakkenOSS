import { describe, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryFormProvider } from '../../../../contexts/QueryFormContext';
import {
  AddValue,
  Condition,
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';
import ConditionList from '.';
import userEvent from '@testing-library/user-event';
import React from 'react';

describe('ConditionList', () => {
  let setFocusedInput: ReturnType<typeof vi.fn>;
  let setIsGuideVisible: ReturnType<typeof vi.fn>;
  let setConditions: ReturnType<typeof vi.fn>;
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

  beforeEach(() => {
    setFocusedInput = vi.fn();
    setIsGuideVisible = vi.fn();
    setConditions = vi.fn();
  });
  it('Should display a list of 3 conditions', () => {
    render(
      <QueryFormProvider>
        <ConditionList
          conditionLengths={{ P: 3, EXISTS: 0 }}
          focusedInput={InputType.CONCEPT}
          setFocusedInput={setFocusedInput}
          isGuideVisible={true}
          conditions={mockConditions}
          conditionType={ConditionType.hypotheses}
          setConditions={setConditions}
          setIsGuideVisible={setIsGuideVisible}
        />
      </QueryFormProvider>,
    );

    const conditions = screen.getAllByTestId('condition-item');
    expect(conditions).toHaveLength(3);
  });
  it('Should display the name of the label', () => {
    render(
      <QueryFormProvider>
        <ConditionList
          conditionLengths={{ P: 3, EXISTS: 0 }}
          focusedInput={InputType.CONCEPT}
          setFocusedInput={setFocusedInput}
          isGuideVisible={true}
          conditions={mockConditions}
          conditionType={ConditionType.hypotheses}
          setConditions={setConditions}
          setIsGuideVisible={setIsGuideVisible}
        />
      </QueryFormProvider>,
    );

    const firstCondition = screen.getByText((content) =>
      content.includes('234'),
    );
    expect(firstCondition).toBeInTheDocument();
  });
  it('Should display an empty conditions text if there are no conditions', () => {
    const emptyConditions: Record<number, Condition> = {};

    render(
      <QueryFormProvider>
        <ConditionList
          conditionLengths={{ P: 0, EXISTS: 0 }}
          focusedInput={InputType.CONCEPT}
          setFocusedInput={setFocusedInput}
          isGuideVisible={true}
          conditions={emptyConditions}
          conditionType={ConditionType.hypotheses}
          setConditions={setConditions}
          setIsGuideVisible={setIsGuideVisible}
        />
      </QueryFormProvider>,
    );

    // Use queryAllByTestId instead of getAllByTestId
    const conditions = screen.queryAllByTestId('condition-item');
    expect(conditions).toHaveLength(0);

    expect(
      screen.getByText((content) => content.includes('There are no')),
    ).toBeInTheDocument();
  });
  it('Should display an inline form when pressing the edit icon', async () => {
    render(
      <QueryFormProvider>
        <ConditionList
          conditionLengths={{ P: 3, EXISTS: 0 }}
          focusedInput={InputType.CONCEPT}
          setFocusedInput={setFocusedInput}
          isGuideVisible={true}
          conditions={mockConditions}
          conditionType={ConditionType.hypotheses}
          setConditions={setConditions}
          setIsGuideVisible={setIsGuideVisible}
        />
      </QueryFormProvider>,
    );
    const editButton = screen.getByTestId('edit-icon-1');
    const user = userEvent.setup();
    await user.click(editButton);
    expect(screen.getByLabelText('Select a domain')).toBeInTheDocument();
  });
  it('duplicates a condition', async () => {
    function TestWrapper() {
      const [conditions, setConditions] = React.useState(mockConditions);
      return (
        <QueryFormProvider>
          <ConditionList
            conditionLengths={{ P: 0, EXISTS: 3 }}
            focusedInput={InputType.CONCEPT}
            setFocusedInput={setFocusedInput}
            isGuideVisible={true}
            conditions={conditions}
            conditionType={ConditionType.constraints}
            setConditions={setConditions}
            setIsGuideVisible={setIsGuideVisible}
          />
        </QueryFormProvider>
      );
    }

    render(<TestWrapper />);

    const user = userEvent.setup();
    const duplicateButton = screen.getByTestId('duplicate-icon-0');
    const conditionToDuplicate = screen.getAllByText((content) =>
      content.includes('123'),
    );
    expect(conditionToDuplicate).toHaveLength(1);
    await user.click(duplicateButton);
    const duplicatedConditions = screen.getAllByText((content) =>
      content.includes('123'),
    );
    expect(duplicatedConditions).toHaveLength(2);
  });
  it('deletes a condition', async () => {
    function TestWrapper() {
      const [conditions, setConditions] = React.useState(mockConditions);
      return (
        <QueryFormProvider>
          <ConditionList
            conditionLengths={{ P: 0, EXISTS: 3 }}
            focusedInput={InputType.CONCEPT}
            setFocusedInput={setFocusedInput}
            isGuideVisible={true}
            conditions={conditions}
            conditionType={ConditionType.constraints}
            setConditions={setConditions}
            setIsGuideVisible={setIsGuideVisible}
          />
        </QueryFormProvider>
      );
    }

    render(<TestWrapper />);

    const user = userEvent.setup();
    const deleteIcon = screen.getByTestId('delete-icon-0');
    const conditionToDelete = screen.getByText((content) =>
      content.includes('123'),
    );
    expect(conditionToDelete).toBeInTheDocument();
    await user.click(deleteIcon);
    expect(conditionToDelete).not.toBeInTheDocument();
  });
});
