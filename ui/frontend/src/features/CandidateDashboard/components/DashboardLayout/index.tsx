import './index.css';
import { useState } from 'react';
import { QueryCandidatesTable } from '../CandidatesTable';
import { CandidateResultType } from '../../types';
import { SearchedParameters } from '../../../../contexts/QueryContext/types';
import { defaultConfig } from '../../../../configI';
import { QueryCandidatePage } from '../CandidatePage/Query';
import { ExplanationProvider } from '../../../../contexts/ExplanationContext';
import { QueryCandidatePageSkeleton } from '../../../../shared/components/LoadingSkeletons';

interface DashboardLayoutProps {
  searchedParameters: SearchedParameters | undefined;
  candidatesResult: CandidateResultType[] | undefined;
}

const DashboardLayout = ({
  searchedParameters,
  candidatesResult,
}: DashboardLayoutProps) => {
  if (!candidatesResult)
    return (
      <section className="results-summary">
        <QueryCandidatePageSkeleton />
      </section>
    );

  const [currentPage, setCurrentPage] = useState<number>(1);
  const candidatesPerPage = defaultConfig.candidatesNumber;
  const totalPages = Math.ceil(candidatesResult.length / candidatesPerPage);
  const [candidatesToShow, setCandidatesToShow] = useState<
    CandidateResultType[]
  >(
    Array.from(
      candidatesResult.slice(
        (currentPage - 1) * candidatesPerPage,
        candidatesPerPage * currentPage,
      ),
    ),
  );
  const [selectedCandidate, setSelectedCandidate] =
    useState<CandidateResultType>(candidatesToShow[0]);

  const handleChangePage = (pageNumber: number) => {
    setCurrentPage(pageNumber);
    setCandidatesToShow(
      candidatesResult.slice(
        (pageNumber - 1) * candidatesPerPage,
        candidatesPerPage * pageNumber,
      ),
    );
  };

  if (!candidatesResult.length) {
    return <p className="empty-results">We could not find any result</p>;
  }

  return (
    <ExplanationProvider>
      {searchedParameters && (
        <section className="results-summary">
          <QueryCandidatesTable
            candidates={candidatesToShow}
            setCandidates={setCandidatesToShow}
            searchedParameters={searchedParameters}
            setSelectedCandidate={setSelectedCandidate}
            selectedCandidate={selectedCandidate}
            totalPages={totalPages}
            currentPage={currentPage}
            handleChangePage={handleChangePage}
          />
          <QueryCandidatePage
            selectedCandidate={selectedCandidate}
            searchedParameters={searchedParameters}
          />
        </section>
      )}
    </ExplanationProvider>
  );
};

export default DashboardLayout;
