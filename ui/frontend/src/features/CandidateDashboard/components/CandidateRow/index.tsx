import './index.css';
import { TableCell, TableRow } from '@mui/material';
import { CandidateResultType } from '../../types';
import React, { SetStateAction } from 'react';
import ScoreMarker from '../../../../shared/components/ScoreMarker';
import { SearchedParameters } from '../../../../contexts/QueryContext/types';
import { useExplanationContext } from '../../../../contexts/ExplanationContext';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import { makeTripleStringArr } from '../utils';

interface QueryCandidateRowProps {
  candidate: CandidateResultType;
  searchedParameters: SearchedParameters;
  selectedCandidate: CandidateResultType;
  setSelectedCandidate: React.Dispatch<SetStateAction<CandidateResultType>>;
}

export const QueryCandidateRow = ({
  candidate,
  selectedCandidate,
  setSelectedCandidate,
}: QueryCandidateRowProps) => {
  const { explanations } = useExplanationContext();
  const explanationsRequested = [...explanations.entries()].filter(([key]) =>
    key.includes(candidate.variableAssignments.X),
  );
  const lastExplanationRequested =
    explanationsRequested[explanationsRequested.length - 1];

  return (
    <>
      <TableRow
        onClick={() => setSelectedCandidate(candidate)}
        className={`${selectedCandidate === candidate ? 'active' : ''} candidate-row`}
        sx={{ position: 'relative' }}
        data-testid="candidate-row"
      >
        <TableCell sx={{ maxWidth: '180px' }}>
          <em className="candidate-name">{candidate.name}</em>
          <p className="candidate-domain">{`${candidate.domain?.replace(/_/g, ' ')}`}</p>
        </TableCell>
        <TableCell>
          {candidate.queryScore.toFixed(5)}{' '}
          <ScoreMarker
            score={Number(candidate.queryScore.toFixed(5))}
            size="small"
          />
        </TableCell>
        {lastExplanationRequested && (
          <TableCell
            sx={{ padding: 0, margin: 0 }}
            className={`${explanations.get(lastExplanationRequested[0])?.status} explanation-status`}
          ></TableCell>
        )}
      </TableRow>
    </>
  );
};

interface ValidationCandidateRowProps {
  triple: ScoredTriple;
  selectedTriple: ScoredTriple;
  setSelectedTriple: React.Dispatch<SetStateAction<ScoredTriple>>;
}

export const ValidationCandidateRow = ({
  triple,
  selectedTriple,
  setSelectedTriple,
}: ValidationCandidateRowProps) => {
  const tripleStrId = makeTripleStringArr(selectedTriple.triple, 'id');
  const tripleStrName = makeTripleStringArr(selectedTriple.triple, 'name');
  const { explanations } = useExplanationContext();
  const explanationsRequested = [...explanations.entries()].filter(([key]) =>
    key.includes(tripleStrId.join('-')),
  );
  const lastExplanationRequested =
    explanationsRequested[explanationsRequested.length - 1];

  return (
    <>
      <TableRow
        onClick={() => setSelectedTriple(triple)}
        className={`${selectedTriple === triple ? 'active' : ''} candidate-row`}
        sx={{ position: 'relative' }}
      >
        <TableCell sx={{ maxWidth: '180px' }}>
          {tripleStrName.join('-')}
        </TableCell>
        <TableCell>
          {triple.score?.toFixed(5)}{' '}
          <ScoreMarker score={Number(triple.score?.toFixed(5))} size="small" />
        </TableCell>
        {lastExplanationRequested && (
          <TableCell
            sx={{ padding: 0, margin: 0 }}
            className={`${explanations.get(lastExplanationRequested[0])?.status} explanation-status`}
          ></TableCell>
        )}
      </TableRow>
    </>
  );
};

export default { QueryCandidateRow, ValidationCandidateRow };
