import './index.css';
import { useQueryContext } from '../../contexts/QueryContext';
import editIcon from '../../assets/images/icons/edit-regular.svg';
import { ThemeProvider, createTheme } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import DashboardLayout from '../../features/CandidateDashboard/components/DashboardLayout';
import { useEffect, useState } from 'react';
import {
  getNodeNames,
  getTripleFromCondition,
} from '../../features/QueryForm/components/MainForm/utils';
import { CandidateResultType } from '../../features/CandidateDashboard/types';

const theme = createTheme({
  typography: {
    allVariants: {
      fontFamily: 'arial-nova',
      textTransform: 'none',
    },
  },
});

const ResultsPage = () => {
  const navigate = useNavigate();
  const { searchedParameters, candidatesResult } = useQueryContext();
  if (!searchedParameters) return;

  const [candidates, setCandidates] = useState<CandidateResultType[]>();

  useEffect(() => {
    const { variable } = getTripleFromCondition(
      Object.values(searchedParameters?.hypotheses)[0],
    );

    const fetchNames = async () => {
      const nodeIds = candidatesResult.candidates.map(
        (candidate) => candidate.variableAssignments.X,
      );
      const nodeNames: Record<string, string> = await getNodeNames(nodeIds);
      const namedCandidates = candidatesResult.candidates.map((candidate) => {
        return {
          ...candidate,
          name: nodeNames[candidate.variableAssignments.X] ?? 'unknown',
          domain: variable.domain,
        };
      });
      setCandidates(namedCandidates);
    };
    fetchNames();
  }, []);

  const handleEditQuery = () => {
    navigate('/');
  };

  return (
    <ThemeProvider theme={theme}>
      <section className="results-section">
        <div className="results-main-info-wrapper">
          <div className="results-main-info">
            <h2>Search Results</h2>
            <div>
              <div className="query-formula">
                <u>QUERY FORMULA:</u>
                <span>{searchedParameters?.query}</span>
                <img
                  src={editIcon}
                  onClick={handleEditQuery}
                  alt="edit query icon"
                ></img>
              </div>
            </div>
          </div>
          <div className="results-display-wrapper">
            <DashboardLayout
              candidatesResult={candidates}
              searchedParameters={searchedParameters}
            />
          </div>
        </div>
      </section>
    </ThemeProvider>
  );
};

export default ResultsPage;
