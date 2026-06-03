/*Validation form inputs*/

import {
  Autocomplete,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
  TextField,
} from '@mui/material';
import { Concept } from '../../../../static/datasets';
import { updateConcepts } from '../../../../features/QueryForm/components/MainForm/utils';
import React, { SetStateAction, useEffect, useMemo, useState } from 'react';
import { createDebouncedSearch } from '../../../../features/QueryForm/utils';
import { SelectedOptions } from '../ValidationForm';

interface EntityInputProps {
  position: 'subject' | 'object';
  possibleDomains: string[];
  setError: React.Dispatch<SetStateAction<string>>;
  selectedOptions: SelectedOptions;
  setSelectedOptions: React.Dispatch<SetStateAction<SelectedOptions>>;
}

interface RelationInputProps {
  possibleRelations: string[];
  setSelectedOptions: React.Dispatch<SetStateAction<SelectedOptions>>;
  selectedOptions: SelectedOptions;
}

export const EntityInputs = ({
  position,
  possibleDomains,
  setError,
  selectedOptions,
  setSelectedOptions,
}: EntityInputProps) => {
  const [possibleConcepts, setPossibleConcepts] = useState<Concept[]>([]);
  const selectedDomain =
    position === 'subject'
      ? selectedOptions.subjectDomain
      : selectedOptions.objectDomain;

  const [loadingConcepts, setLoadingConcepts] = useState<boolean>(true);

  const emptyConcept: Concept = {
    domain: '',
    id: '',
    name: '',
  };

  const [inputValue, setInputValue] = useState('');

  useEffect(() => {
    setSelectedOptions((prev) =>
      position === 'subject'
        ? { ...prev, subjectConcept: emptyConcept }
        : { ...prev, objectConcept: emptyConcept },
    );

    const filterConcepts = async () => {
      if (selectedDomain) {
        setLoadingConcepts(true);
        try {
          const filteredConcepts = await updateConcepts(selectedDomain);
          setPossibleConcepts(filteredConcepts);
        } catch (err) {
          console.error('Failed to update concepts', err);
          setError('Failed to update concepts');
        } finally {
          setLoadingConcepts(false);
        }
      }
    };

    filterConcepts();
  }, [selectedDomain]);

  const handleConceptSearch = useMemo(
    () =>
      createDebouncedSearch(
        400,
        selectedDomain,
        setPossibleConcepts,
        setLoadingConcepts,
        false,
      ),
    [selectedDomain],
  );

  const selectedConcept =
    position === 'subject'
      ? selectedOptions.subjectConcept
      : selectedOptions.objectConcept;

  return (
    <div className="input-wrapper concept">
      <FormControl size="small" className="entity-inputs">
        <InputLabel
          id={`${position}-domain-label`}
        >{`${position} domain`}</InputLabel>
        <Select
          labelId={`${position}-domain-label`}
          label={`${position} domain`}
          value={selectedDomain}
          onChange={(e: SelectChangeEvent<string>) =>
            setSelectedOptions((prev) =>
              position === 'subject'
                ? { ...prev, subjectDomain: e.target.value }
                : { ...prev, objectDomain: e.target.value },
            )
          }
          size="small"
          className="input"
        >
          <MenuItem value="">--</MenuItem>
          {possibleDomains.map((domain, index) => (
            <MenuItem key={index} value={domain}>
              {domain}
            </MenuItem>
          ))}
        </Select>

        <Autocomplete<Concept>
          loading={loadingConcepts}
          loadingText={
            selectedDomain ? 'Loading...' : 'Select a domain to load entities'
          }
          options={possibleConcepts}
          value={selectedConcept ?? null}
          inputValue={inputValue}
          onChange={(_, newValue) => {
            setSelectedOptions((prev) =>
              position === 'subject'
                ? { ...prev, subjectConcept: newValue ?? emptyConcept }
                : { ...prev, objectConcept: newValue ?? emptyConcept },
            );
          }}
          getOptionLabel={(option) => option.name}
          isOptionEqualToValue={(a, b) => a.id === b.id}
          onInputChange={(_, newValue) => {
            setInputValue(newValue);
            handleConceptSearch(newValue);
          }}
          size="small"
          className="input"
          renderInput={(params) => (
            <TextField {...params} label={`${position} entity`} />
          )}
        />
      </FormControl>
    </div>
  );
};

export const RelationInput = ({
  possibleRelations,
  setSelectedOptions,
  selectedOptions,
}: RelationInputProps) => {
  return (
    <div className="input-wrapper relation">
      <FormControl size="small">
        <InputLabel id="relation-select-label">relation</InputLabel>
        <Select
          labelId="relation-select-label"
          label="relation"
          value={selectedOptions.relation}
          onChange={(e: SelectChangeEvent<string>) =>
            setSelectedOptions((prev) => ({
              ...prev,
              relation: e.target.value,
            }))
          }
          size="small"
          className="input"
        >
          {Object.values(possibleRelations).length === 0 ? (
            <MenuItem>{'No possible relations'}</MenuItem>
          ) : (
            [
              <MenuItem key={'none'} value="">
                --
              </MenuItem>,
              <MenuItem key={'all'} value="ANY">
                <strong>ANY</strong>
              </MenuItem>,
              Array.from(new Set(possibleRelations)).map((relation) => (
                <MenuItem key={relation} value={relation}>
                  {relation.replace(/_/g, ' ')}
                </MenuItem>
              )),
            ]
          )}
        </Select>
      </FormControl>
    </div>
  );
};
