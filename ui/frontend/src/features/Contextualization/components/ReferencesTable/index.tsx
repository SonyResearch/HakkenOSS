import {
  Box,
  Button,
  Collapse,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  ContextualizationResult,
  Filters,
  Sorting,
} from '../../../CandidateDashboard/types';
import { ReferenceSkeleton } from '../../../../shared/components/LoadingSkeletons';
import React, {
  SetStateAction,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { ReferenceFilters } from '../../../ReferencesFilters';
import { ReferenceRow } from './ReferenceRow';
import { useMobile } from '../../../../hooks/useMobile';
import ScatterPlotIcon from '../../../../shared/components/ScatterPlotIcon';
import ContextualizationScatterPlot from '../../../Visualizations/ScatterPlot';

const ReferencesTable = ({
  contextualization,
  error,
  loading,
  filters,
  sorting,
  setFilters,
  setSorting,
}: {
  contextualization: ContextualizationResult | undefined;
  error: string;
  loading: boolean;
  filters: Filters;
  sorting: Sorting;
  setFilters: React.Dispatch<SetStateAction<Filters>>;
  setSorting: React.Dispatch<SetStateAction<Sorting>>;
}) => {
  const isMobile = useMobile();
  const [showFilters, setShowFilters] = useState<boolean>(!isMobile);
  const [showScatterPlot, setShowScatterPlot] = useState<boolean>(false);
  const scatterPlotRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const clickedVisualization =
        scatterPlotRef.current &&
        scatterPlotRef.current.contains(event.target as Node);
      if (!clickedVisualization) {
        setShowScatterPlot(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  const filteredAndSortedReferences = useMemo(() => {
    const references = contextualization?.references ?? [];

    const filtered = references.filter((reference) => {
      const matchesTitle = reference.publication_info.title
        .toLowerCase()
        .includes(filters.title.toLowerCase());

      const matchesAuthor = filters.author
        ? reference.publication_info.authors
            .map((a) => `${a.first_name} ${a.last_name}`)
            .join(', ')
            .toLowerCase()
            .includes(filters.author.toLowerCase())
        : true;

      const matchesAbstract = filters.abstract
        ? reference.publication_info.abstract
          ? reference.publication_info.abstract
              .toLowerCase()
              .includes(filters.abstract.toLowerCase())
          : false
        : true;

      return matchesTitle && matchesAuthor && matchesAbstract;
    });

    return filtered.sort((a, b) => {
      const aValue =
        sorting.category === 'score'
          ? Number(a.score)
          : Number(a.publication_info[sorting.category]);
      const bValue =
        sorting.category === 'score'
          ? Number(b.score)
          : Number(b.publication_info[sorting.category]);

      return sorting.order === 'ascending' ? aValue - bValue : bValue - aValue;
    });
  }, [contextualization?.references, sorting, filters]);

  if (error || (contextualization && !contextualization?.references.length)) {
    const emptyResult =
      error === 'Empty contextualization result' ||
      !contextualization?.references.length;

    return (
      <div className="contextualization-error">
        {emptyResult ? (
          <p>We couldn&apos;t find any references for this triple</p>
        ) : (
          <p>Something went wrong getting contextualization</p>
        )}
      </div>
    );
  }

  return (
    <Box>
      <h3>
        <u>References</u>
        <div
          onClick={() => setShowScatterPlot(!showScatterPlot)}
          className="scatter-plot-button"
        >
          <ScatterPlotIcon size={64} />
        </div>
      </h3>
      {contextualization?.summary && (
        <div className="reference-summary">
          <h4>Summary</h4>
          <p>{contextualization.summary}</p>
        </div>
      )}
      {contextualization && (
        <div
          className={`scatter-plot-modal ${showScatterPlot ? 'visible' : ''}`}
        >
          <ContextualizationScatterPlot
            ref={scatterPlotRef}
            contextualization={contextualization}
            handleReferenceClick={() => setShowScatterPlot(false)}
          />
        </div>
      )}
      {isMobile && (
        <Button
          sx={{
            marginBottom: '0.5rem',
            width: '100%',
            border: '1px solid gray',
            color: 'gray',
          }}
          size="small"
          variant="text"
          onClick={() => setShowFilters(!showFilters)}
        >
          {showFilters ? 'Hide Filters' : 'Show Filters'}
        </Button>
      )}
      <Collapse in={showFilters}>
        <ReferenceFilters
          setSorting={setSorting}
          setFilters={setFilters}
          sorting={sorting}
          filters={filters}
        />
      </Collapse>
      <TableContainer
        sx={{ maxHeight: '100vh', overflow: 'scroll', scrollbarWidth: 'thin' }}
      >
        <Table stickyHeader sx={{ width: '100%' }}>
          <TableHead sx={{ width: '100%' }}>
            <TableRow>
              <TableCell sx={{ width: isMobile ? '90%' : '35%' }}>
                Title
              </TableCell>
              <TableCell
                sx={{ width: '30%', display: isMobile ? 'none' : 'table-cell' }}
              >
                Authors
              </TableCell>
              <TableCell
                sx={{ width: '10%', display: isMobile ? 'none' : 'table-cell' }}
              >
                Year
              </TableCell>
              <TableCell
                sx={{ width: '10%', display: isMobile ? 'none' : 'table-cell' }}
              >
                Citations
              </TableCell>
              <TableCell
                sx={{ width: '10%', display: isMobile ? 'none' : 'table-cell' }}
              >
                Score
              </TableCell>
              <TableCell></TableCell>
            </TableRow>
          </TableHead>

          {loading ? (
            <ReferenceSkeleton candidatesNumber={20} />
          ) : filteredAndSortedReferences.length === 0 ? (
            <TableBody>
              <TableRow>
                <TableCell colSpan={5}>
                  No references found for this parameters
                </TableCell>
              </TableRow>
            </TableBody>
          ) : (
            <TableBody>
              {filteredAndSortedReferences.map((reference, index) => (
                <ReferenceRow
                  reference={reference}
                  key={index}
                  titleFilter={filters.title}
                  authorFilter={filters.author}
                  abstractFilter={filters.abstract}
                />
              ))}
            </TableBody>
          )}
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ReferencesTable;
