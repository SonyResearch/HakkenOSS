import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import ResultsPage from '.';
import { useQueryContext } from '../../contexts/QueryContext';
import { useNavigate } from 'react-router-dom';
import {
  getConditionText,
  getNodeNames,
  getTripleFromCondition,
} from '../../features/QueryForm/components/MainForm/utils';
import {
  AddValue,
  Condition,
  ConditionType,
} from '../../contexts/QueryContext/types';

vi.mock('../../contexts/QueryContext');
vi.mock('react-router-dom', () => ({
  ...vi.importActual('react-router-dom'),
  useNavigate: vi.fn(),
}));
vi.mock('../../features/QueryForm/components/MainForm/utils', () => ({
  getNodeNames: vi.fn(),
  getTripleFromCondition: vi.fn(),
  getConditionText: vi.fn(),
}));

const mockUseQueryContext = useQueryContext as unknown as ReturnType<
  typeof vi.fn
>;
const mockUseNavigate = useNavigate as unknown as ReturnType<typeof vi.fn>;
const mockGetNodeNames = getNodeNames as unknown as ReturnType<typeof vi.fn>;
const mockGetTripleFromCondition =
  getTripleFromCondition as unknown as ReturnType<typeof vi.fn>;
const mockGetConditionText = getConditionText as unknown as ReturnType<
  typeof vi.fn
>;
describe('ResultsPage', () => {
  const mockNavigate = vi.fn();

  const mockCandidate = (name: string, score: number) => ({
    variableAssignments: { X: name },
    queryScore: score,
    conditionsScores: {},
  });

  const mockCandidates = [
    mockCandidate('Ibuprofen', 9.123),
    mockCandidate('Paracetamol', 8.321),
    mockCandidate('Aspirin', 7),
  ];

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
    mockUseNavigate.mockReturnValue(mockNavigate);

    mockUseQueryContext.mockReturnValue({
      searchedParameters: {
        query: 'X, TREATS, B',
        hypotheses: mockConditions,
        constraints: mockConditions,
      },
      candidatesResult: {
        candidates: mockCandidates,
      },
    });

    mockGetTripleFromCondition.mockReturnValue({
      variable: { domain: 'CHEMICAL', id: '123' },
      relation: 'TREATS',
      concept: { domain: 'DISEASE', id: '234' },
      isSubjectPrediction: true,
    });

    mockGetConditionText.mockReturnValue('Ibuprofen, TREATS, X');

    mockGetNodeNames.mockResolvedValue({
      node1: 'Ibuprofen',
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders search results header', async () => {
    render(<ResultsPage />);
    expect(screen.getByText('Search Results')).toBeInTheDocument();
  });

  it('renders query formula', async () => {
    render(<ResultsPage />);
    expect(screen.getByText('X, TREATS, B')).toBeInTheDocument();
  });

  it('calls getNodeNames and sets candidates', async () => {
    render(<ResultsPage />);
    mockGetConditionText.mockReturnValue('Ibuprofen, TREATS, X');

    await waitFor(() => {
      expect(mockGetNodeNames).toHaveBeenCalledWith([
        'Ibuprofen',
        'Paracetamol',
        'Aspirin',
      ]);
    });
  });

  it('navigates when edit icon is clicked', async () => {
    render(<ResultsPage />);
    const icon = screen.getByAltText('edit query icon');
    await userEvent.click(icon);
    expect(mockNavigate).toHaveBeenCalledWith('/');
  });

  it('returns nothing if searchedParameters is missing', () => {
    mockUseQueryContext.mockReturnValue({
      searchedParameters: null,
      candidatesResult: null,
    });

    const { container } = render(<ResultsPage />);
    expect(container.firstChild).toBeNull();
  });
});
