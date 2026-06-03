/* Input to select multiple concept names for same domain entities, allows for multiple-condition queries, can only be used in queries using complex query module */

import {
  Autocomplete,
  Box,
  Checkbox,
  Popper,
  PopperProps,
  TextField,
} from '@mui/material';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import React, {
  HTMLAttributes,
  ReactNode,
  SetStateAction,
  useState,
} from 'react';
import {
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';

type ConceptOption = {
  name: string;
  selected: boolean;
};

interface MultipleConceptSelectorProps {
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  conditionType: ConditionType;
  handleConceptSearch: (query: string) => void;
  possibleConceptNames: string[];
  loadingConcepts: boolean;
  noOptionsText: string;
}

export const MultipleConceptSelector = ({
  setFocusedInput,
  conditionType,
  handleConceptSearch,
  possibleConceptNames,
  loadingConcepts,
  noOptionsText,
}: MultipleConceptSelectorProps) => {
  const { state, dispatch } = useQueryFormContext();
  const [search, setSearch] = useState<string>('');
  const allConcepts =
    state[conditionType].form.tailValue[0] !== ''
      ? [
          ...new Set([
            ...possibleConceptNames,
            ...state[conditionType].form.tailValue,
          ]),
        ]
      : possibleConceptNames;
  const sortedConcepts = allConcepts.length
    ? allConcepts.map((concept) => ({
        name: concept,
        selected: state[conditionType].form.tailValue.includes(concept),
      }))
    : [];

  const areConceptsSelected = sortedConcepts.some(
    (concept) => concept.selected,
  );

  const handleInputChange = (query: string) => {
    setSearch(query);
    if (conditionType === ConditionType.hypotheses) {
      handleConceptSearch(query);
    }
  };

  const CustomConceptSeparator = React.forwardRef(function ConceptDropDown(
    props: HTMLAttributes<HTMLElement>,
    ref,
  ) {
    const { children, ...other } = props;
    const selectedConcepts: ReactNode[] = [];
    const unSelectedConcepts: ReactNode[] = [];

    React.Children.forEach(children as React.ReactElement[], (child) => {
      const isChecked =
        child?.props?.children[0].props.children[0].props.checked;
      if (isChecked === false) unSelectedConcepts.push(child);
      else if (isChecked === true) selectedConcepts.push(child);
    });

    return (
      <Box
        ref={ref}
        {...other}
        sx={{
          display: 'grid',
          gridTemplateColumns: selectedConcepts.length > 0 ? '1fr 1fr' : '1fr',
        }}
      >
        <Box sx={{ width: '220px', fontSize: '0.8rem' }}>
          {selectedConcepts}
        </Box>
        <Box sx={{ width: '220px', fontSize: '0.8rem' }}>
          {unSelectedConcepts}
        </Box>
      </Box>
    );
  });

  const CustomPopper = (props: PopperProps) => {
    return (
      <Popper
        {...props}
        placement="bottom-start"
        style={{ maxWidth: areConceptsSelected ? 600 : 300 }}
      ></Popper>
    );
  };

  const handleTailValueChange = (
    event: React.SyntheticEvent,
    selectedConcepts: ConceptOption | ConceptOption[] | null,
  ) => {
    if (selectedConcepts) {
      const concepts = Array.isArray(selectedConcepts)
        ? selectedConcepts.map((selectedConcept) => selectedConcept.name)
        : [selectedConcepts.name];
      const newConcept = concepts[concepts.length];
      const value = state[conditionType].form.tailValue.includes(newConcept)
        ? state[conditionType].form.tailValue.filter(
            (option) => option !== newConcept,
          )
        : concepts;
      dispatch({
        type: 'UPDATE_FIELD',
        formType: conditionType,
        field: 'tailValue',
        value,
      });
    }
  };

  return (
    <Autocomplete
      noOptionsText={noOptionsText}
      loading={loadingConcepts}
      sx={{
        padding: 0,
        margin: 0,
        minWidth: 'var(--min-input-width)',
        maxWidth: 'var(--max-input-width)',
      }}
      onMouseEnter={() => setFocusedInput(InputType.CONCEPT)}
      onMouseLeave={() => setFocusedInput(null)}
      PopperComponent={CustomPopper}
      value={sortedConcepts.filter((concept) =>
        state[conditionType].form.tailValue.includes(concept.name),
      )}
      options={sortedConcepts}
      groupBy={(option) => (option.selected ? 'selected' : 'unselected')}
      inputValue={search}
      onChange={handleTailValueChange}
      getOptionLabel={(option) => option.name || ''}
      isOptionEqualToValue={(option, value) => option.name === value.name}
      disableCloseOnSelect
      renderGroup={(params) => (
        <React.Fragment key={params.key}>{params.children}</React.Fragment>
      )}
      multiple
      renderOption={(props, option) => {
        return (
          <li {...props}>
            <Checkbox
              sx={{ '&.Mui-checked': { color: 'var(--blue)' } }}
              checked={option.selected}
            />
            {option.name}
          </li>
        );
      }}
      ListboxComponent={CustomConceptSeparator}
      renderInput={(params) => (
        <TextField
          placeholder={areConceptsSelected ? '' : 'lung cancer'}
          {...params}
          onChange={(e) => handleInputChange(e.target.value)}
        />
      )}
    />
  );
};

export default MultipleConceptSelector;
