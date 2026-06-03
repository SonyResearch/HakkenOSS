import { Clear } from '@mui/icons-material';
import {
  FormControl,
  FormLabel,
  Stack,
  Select,
  MenuItem,
  TextField,
  Button,
  IconButton,
  InputAdornment,
} from '@mui/material';
import {
  Filters,
  Sorting,
  SortingCategories,
} from '../CandidateDashboard/types';
import React, { SetStateAction } from 'react';
import { useMobile } from '../../hooks/useMobile';

interface ReferenceFiltersProps {
  sorting: Sorting;
  setSorting: React.Dispatch<SetStateAction<Sorting>>;
  filters: Filters;
  setFilters: React.Dispatch<SetStateAction<Filters>>;
}
export const ReferenceFilters = ({
  sorting,
  setSorting,
  filters,
  setFilters,
}: ReferenceFiltersProps) => {
  const isMobile = useMobile();
  const handleClearFilters = () => {
    setFilters({ author: '', title: '', abstract: '' });
    setSorting({ category: 'score', order: 'ascending' });
  };

  return (
    <Stack
      className="reference-filters"
      direction={'row'}
      gap="3rem"
      alignItems={'flex-end'}
      justifyContent={'center'}
      sx={{ padding: '0.5rem 0', borderRadius: '10px', width: 'fit-content' }}
    >
      <FormControl>
        <FormLabel>Sort by</FormLabel>
        <Stack className="sort" direction={'row'} gap="1rem">
          <Select
            MenuProps={{
              disablePortal: true,
            }}
            sx={{ backgroundColor: 'white' }}
            size="small"
            onChange={(e) =>
              setSorting((prev) => ({
                ...prev,
                category: e.target.value as SortingCategories,
              }))
            }
            value={sorting.category}
          >
            <MenuItem value="year">Year</MenuItem>
            <MenuItem value="citations_count">Citation Count</MenuItem>
            <MenuItem value="score">Score</MenuItem>
          </Select>

          <Select
            MenuProps={{
              disablePortal: true,
            }}
            sx={{ backgroundColor: 'white', width: isMobile ? '100%' : 'auto' }}
            size="small"
            onChange={(e) =>
              setSorting((prev) => ({
                ...prev,
                order: e.target.value as 'ascending' | 'descending',
              }))
            }
            value={sorting.order}
          >
            <MenuItem value="ascending">Ascending</MenuItem>
            <MenuItem value="descending">Descending</MenuItem>
          </Select>
        </Stack>
      </FormControl>

      <TextField
        label="Filter by Author"
        size="small"
        sx={{ width: isMobile ? '100%' : 'auto' }}
        value={filters.author}
        onChange={(e) =>
          setFilters((prev) => ({ ...prev, author: e.target.value }))
        }
        InputProps={{
          endAdornment: filters.author ? (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={() => setFilters((prev) => ({ ...prev, author: '' }))}
              >
                <Clear fontSize="small" />
              </IconButton>
            </InputAdornment>
          ) : null,
        }}
      />

      <TextField
        label="Filter by Title"
        sx={{ width: isMobile ? '100%' : 'auto' }}
        size="small"
        value={filters.title}
        onChange={(e) =>
          setFilters((prev) => ({ ...prev, title: e.target.value }))
        }
        InputProps={{
          endAdornment: filters.title ? (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={() => setFilters((prev) => ({ ...prev, title: '' }))}
              >
                <Clear fontSize="small" />
              </IconButton>
            </InputAdornment>
          ) : null,
        }}
      />

      <TextField
        label="Filter by Abstract Text"
        sx={{ width: isMobile ? '100%' : 'auto' }}
        size="small"
        value={filters.abstract}
        onChange={(e) =>
          setFilters((prev) => ({ ...prev, abstract: e.target.value }))
        }
        InputProps={{
          endAdornment: filters.abstract ? (
            <InputAdornment position="end">
              <IconButton
                size="small"
                onClick={() =>
                  setFilters((prev) => ({ ...prev, abstract: '' }))
                }
              >
                <Clear fontSize="small" />
              </IconButton>
            </InputAdornment>
          ) : null,
        }}
      />
      <Button onClick={handleClearFilters}>Clear</Button>
    </Stack>
  );
};
