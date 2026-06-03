import './index.css';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from '@mui/material';
import { CandidateResultType } from '../../types';
import { SetStateAction, useState } from 'react';
import { SearchedParameters } from '../../../../contexts/QueryContext/types';
import { QueryCandidateRow } from '../CandidateRow';
import { Pagination } from '../../../../shared/components/Pagination';

interface QueryCandidatesTableProps {
  candidates: CandidateResultType[];
  setCandidates: React.Dispatch<SetStateAction<CandidateResultType[]>>;
  searchedParameters: SearchedParameters;
  setSelectedCandidate: React.Dispatch<SetStateAction<CandidateResultType>>;
  selectedCandidate: CandidateResultType;
  totalPages: number;
  currentPage: number;
  handleChangePage: (page: number) => void;
}

const QueryCandidatesTable = ({
  candidates,
  setCandidates,
  searchedParameters,
  setSelectedCandidate,
  selectedCandidate,
  totalPages,
  currentPage,
  handleChangePage,
}: QueryCandidatesTableProps) => {
  const [isDescendingOrder, setIsDescendingOrder] = useState<boolean>(false);
  const toggleCandidatesOrder = () => {
    setCandidates(candidates.reverse());
    setIsDescendingOrder(!isDescendingOrder);
  };
  return (
    <div className="candidate-wrapper">
      <h3>
        Candidates{' '}
        <span className="candidates-found">({candidates.length} found)</span>
      </h3>
      <Table className={'candidates-table'}>
        <TableHead>
          <TableRow>
            <TableCell>NAME</TableCell>
            <TableCell>
              {' '}
              CONFIDENCE SCORE{' '}
              <span onClick={toggleCandidatesOrder}>
                {!isDescendingOrder ? '\u25BC' : '\u25B2'}
              </span>
            </TableCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {candidates.map((candidate, index) => (
            <QueryCandidateRow
              key={index}
              candidate={candidate}
              searchedParameters={searchedParameters}
              selectedCandidate={selectedCandidate}
              setSelectedCandidate={setSelectedCandidate}
            />
          ))}
        </TableBody>
      </Table>
      {totalPages > 1 && (
        <Pagination
          numberOfPages={totalPages}
          currentPage={currentPage}
          handleChangePage={handleChangePage}
        />
      )}
    </div>
  );
};

export { QueryCandidatesTable };
