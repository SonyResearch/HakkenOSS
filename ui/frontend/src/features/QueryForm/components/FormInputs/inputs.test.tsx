import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { describe, it, vi } from 'vitest';
import RelationInput from './RelationInput';
import { QueryFormProvider } from '../../../../contexts/QueryFormContext';
import {
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';
import userEvent from '@testing-library/user-event';

type RenderOptions = {
  selectedRelation?: string;
  possibleRelations?: string[];
  loadingRelations?: boolean;
  dispatch?: () => void;
  setFocusedInput?: () => void;
  conditionType?: ConditionType;
};

const defaultConditionType = ConditionType.hypotheses;

const renderComponent = ({
  possibleRelations = ['RELATES_TO'],
  loadingRelations = false,
  setFocusedInput = vi.fn(),
  conditionType = defaultConditionType,
}: RenderOptions = {}) => {
  const TestProvider = ({ children }: { children: React.ReactNode }) => {
    return <QueryFormProvider>{children}</QueryFormProvider>;
  };

  render(
    <TestProvider>
      <RelationInput
        setFocusedInput={setFocusedInput}
        conditionType={conditionType}
        possibleRelations={possibleRelations}
        loadingRelations={loadingRelations}
      />
    </TestProvider>,
  );

  return { setFocusedInput };
};

describe('RelationInput', () => {
  it('renders placeholder when no relation selected', () => {
    renderComponent({ selectedRelation: '' });

    expect(screen.getByTestId('relation-placeholder')).toBeInTheDocument();
  });

  it('shows loading message when loadingRelations is true', async () => {
    const user = userEvent.setup();
    renderComponent({ loadingRelations: true });

    await user.click(screen.getByRole('combobox'));

    expect(await screen.findByText('Loading relations...')).toBeInTheDocument();
  });

  it('shows empty message when no possible relations', async () => {
    const user = userEvent.setup();
    renderComponent({ possibleRelations: [] });

    await user.click(screen.getByRole('combobox'));

    expect(
      await screen.findByText('No possible relations'),
    ).toBeInTheDocument();
  });

  it('renders unique relations only', async () => {
    const user = userEvent.setup();

    renderComponent({
      possibleRelations: ['RELATES_TO', 'RELATES_TO', 'child_of'],
    });

    await user.click(screen.getByRole('combobox'));

    expect(await screen.findAllByText('RELATES TO')).toHaveLength(1);
    expect(await screen.findByText('child of')).toBeInTheDocument();
  });
  it('resets to empty when "--" selected', async () => {
    const user = userEvent.setup();

    renderComponent({
      selectedRelation: 'RELATES_TO',
      possibleRelations: ['RELATES_TO'],
    });

    await user.click(screen.getByRole('combobox'));

    await user.click(await screen.findByText('--'));

    expect(screen.getByTestId('relation-placeholder')).toBeInTheDocument();
  });

  it('calls setFocusedInput on hover enter and leave', async () => {
    const user = userEvent.setup();
    const { setFocusedInput } = renderComponent();

    const select = document.getElementById('relation');
    expect(select).toBeTruthy();

    if (select) {
      await user.hover(select);
      expect(setFocusedInput).toHaveBeenCalledWith(InputType.RELATION);

      await user.unhover(select);
      expect(setFocusedInput).toHaveBeenCalledWith(null);
    }
  });
  it('works correctly with different conditionType', async () => {
    const user = userEvent.setup();
    const conditionType = ConditionType.constraints;

    renderComponent({
      conditionType,
      possibleRelations: ['INHIBITS'],
    });

    await user.click(screen.getByRole('combobox'));
    await user.click(await screen.findByText('INHIBITS'));

    expect(screen.getByText('INHIBITS')).toBeInTheDocument();
  });
});
