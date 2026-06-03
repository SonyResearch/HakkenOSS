import { TableBody, TableCell, TableRow } from '@mui/material';
import './index.css';

export const ReferenceSkeleton = ({
  candidatesNumber,
}: {
  candidatesNumber: number;
}) => {
  const skeletonBars = Array.from({ length: candidatesNumber });
  return (
    <TableBody className="summary-skeleton">
      {skeletonBars.map((_, index) => (
        <TableRow key={index}>
          <TableCell
            colSpan={5}
            key={index}
            className="skeleton-bar"
          ></TableCell>
        </TableRow>
      ))}
    </TableBody>
  );
};

export const QueryCandidatePageSkeleton = () => {
  return (
    <div className="candidate-page-skeleton">
      <div></div>
      <div></div>
    </div>
  );
};
