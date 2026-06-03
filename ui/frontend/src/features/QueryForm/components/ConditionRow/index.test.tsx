import ConditionRow from '.';
import {
  AddValue,
  Condition,
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryFormProvider } from '../../../../contexts/QueryFormContext';
import userEvent from '@testing-library/user-event';

describe('ConditionRow', () => {
  let setConditions: ReturnType<typeof vi.fn>;
  let setFocusedInput: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    setConditions = vi.fn();
    setFocusedInput = vi.fn();
    render(
      <QueryFormProvider>
        <ConditionRow
          canAddMoreConditions={true}
          focusedInput={InputType.CONCEPT}
          isGuideVisible={true}
          setFocusedInput={setFocusedInput}
          setConditions={setConditions}
          conditions={mockConditions}
          condition={Object.values(mockConditions)[0]}
          index={0}
          listing={0}
          isOnlyCondition={false}
          conditionType={ConditionType.hypotheses}
        />
      </QueryFormProvider>,
    );
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
  it('Should display the condition passed as props', () => {
    const conditionText = screen.getByText((content) =>
      content.includes('123'),
    );
    expect(conditionText).toBeInTheDocument();
  });
  it('Should stop displaying the condition once you press the edit button and the pyramid guide should appear', async () => {
    const editButton = screen.getByTestId('edit-icon-0');
    const conditionText = screen.getByText((content) =>
      content.includes('123'),
    );
    const user = userEvent.setup();
    await user.click(editButton);
    const pyramidGuide = screen.getByTestId('svg-guide');
    expect(conditionText).not.toBeInTheDocument();
    expect(pyramidGuide).toBeInTheDocument();
  });
});
