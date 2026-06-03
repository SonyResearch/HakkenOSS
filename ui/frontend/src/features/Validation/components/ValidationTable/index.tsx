/*Table displaying validation results*/

import { Button } from '@mui/material';
import { ScoredTriple } from '../../../../pages/ValidationPage';
import { normalizeScore } from '../../utils';
import AssignmentOutlinedIcon from '@mui/icons-material/AssignmentOutlined';
import HighlightOffOutlinedIcon from '@mui/icons-material/HighlightOffOutlined';
import React, { SetStateAction } from 'react';

interface ValidationTableProps {
  scoredTriples: ScoredTriple[];
  setScoredTriples: React.Dispatch<SetStateAction<ScoredTriple[]>>;
  setSelectedTriple: React.Dispatch<SetStateAction<ScoredTriple | undefined>>;
}
const ValidationTable = ({
  scoredTriples,
  setScoredTriples,
  setSelectedTriple,
}: ValidationTableProps) => {
  const handleDeleteTriple = (index: number) => {
    const newTriples = scoredTriples.filter(
      (_, currIndex) => currIndex !== index,
    );
    setScoredTriples(newTriples);
  };

  return (
    <div className="table-wrapper">
      <table className="validation-table">
        <thead>
          <tr>
            <th>Subject</th>
            <th>Relation</th>
            <th>Object</th>
            <th>Confidence Score</th>
          </tr>
        </thead>
        <tbody>
          {scoredTriples.map((scoredTriple, index) => (
            <tr key={index}>
              <td>{scoredTriple.triple[0].name}</td>
              <td>{scoredTriple.triple[1]}</td>
              <td>{scoredTriple.triple[2].name}</td>
              <td>
                {normalizeScore(scoredTriple.score)}{' '}
                <div className="icon-box">
                  <AssignmentOutlinedIcon
                    onClick={() => setSelectedTriple(scoredTriple)}
                  />
                  <HighlightOffOutlinedIcon
                    onClick={() => handleDeleteTriple(index)}
                    className="delete-button"
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Button
        className="reset-button"
        variant="outlined"
        onClick={() => setScoredTriples([])}
      >
        Reset
      </Button>
    </div>
  );
};

export default ValidationTable;
