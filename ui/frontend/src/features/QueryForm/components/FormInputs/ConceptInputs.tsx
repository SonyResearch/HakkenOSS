/*Input to select domain and name for the prediction target entity*/

import './index.css';
import {
  FormControl,
  Select,
  MenuItem,
  Typography,
  Autocomplete,
  TextField,
} from '@mui/material';
import LightTooltip from '../../../../shared/components/LightToolTip';
import { RefObject, SetStateAction, useMemo } from 'react';
import {
  ConditionType,
  InputType,
  PredictionType,
} from '../../../../contexts/QueryContext/types';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import React from 'react';
import MultipleConceptSelector from './MultipleConceptSelector';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { createDebouncedSearch } from '../../utils';

interface ConceptInputProps {
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  conditionType: ConditionType;
  stackRef: RefObject<HTMLDivElement>;
  possibleConceptDomains: string[];
  possibleConceptNames: string[];
  loadingConcepts: boolean;
  setLoadingConcepts: React.Dispatch<SetStateAction<boolean>>;
  setPossibleConceptNames: React.Dispatch<SetStateAction<string[]>>;
}
const ConceptInputs = ({
  setFocusedInput,
  conditionType,
  stackRef,
  possibleConceptDomains,
  possibleConceptNames,
  loadingConcepts,
  setLoadingConcepts,
  setPossibleConceptNames,
}: ConceptInputProps) => {
  const { state, dispatch } = useQueryFormContext();
  const { queryMode } = useQueryContext();
  const isSubjectPrediction =
    state[conditionType].form.predictionType === PredictionType.SUBJECT;
  const noOptionsText = state[conditionType].form.selectedConceptDomain
    ? 'Entity not found'
    : 'Select a domain to load entities';
  const handleTailValueChange = (
    event: React.SyntheticEvent,
    value: string,
  ) => {
    dispatch({
      type: 'UPDATE_FIELD',
      formType: conditionType,
      field: 'tailValue',
      value: [value],
    });
  };
  const handleConceptSearch = useMemo(
    () =>
      createDebouncedSearch(
        400,
        state[conditionType].form.selectedConceptDomain,
        setPossibleConceptNames,
        setLoadingConcepts,
        true,
      ),
    [state[conditionType].form.selectedConceptDomain],
  );

  const allowOnlyOneConcept =
    queryMode === 'simple' || conditionType === ConditionType.constraints;
  return (
    <div className="input-wrapper concept" ref={stackRef}>
      {' '}
      <FormControl
        fullWidth
        sx={{
          minWidth: 'var(--min-input-width)',
          maxWidth: 'var(--max-input-width)',
        }}
      >
        <LightTooltip
          title={`Select a Domain`}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <Select
            onMouseEnter={() => setFocusedInput(InputType.CONCEPT)}
            onMouseLeave={() => setFocusedInput(null)}
            labelId="searchTarget-label"
            id="search-Target"
            value={state[conditionType].form.selectedConceptDomain}
            displayEmpty
            renderValue={(value: string) => {
              if (!value) {
                return (
                  <Typography
                    fontFamily={'arial-nova'}
                    fontWeight={500}
                    className="placeholder"
                    fontSize={'0.7rem'}
                  >
                    {isSubjectPrediction ? 'object domain' : 'subject domain'}
                  </Typography>
                );
              }
              return <span>{value.replace(/_/g, ' ')}</span>;
            }}
            label="domain"
            onChange={(e) =>
              dispatch({
                type: 'UPDATE_CONCEPT_DOMAIN',
                formType: conditionType,
                value: e.target.value,
              })
            }
          >
            {loadingConcepts ? (
              <MenuItem>{'Loading domains...'}</MenuItem>
            ) : (
              [
                <MenuItem key="" value="">
                  --
                </MenuItem>,
                ...(possibleConceptDomains
                  ? possibleConceptDomains.map((conceptDomain, index) => (
                      <MenuItem key={index} value={conceptDomain}>
                        {conceptDomain}
                      </MenuItem>
                    ))
                  : []),
              ]
            )}
          </Select>
        </LightTooltip>
      </FormControl>
      <FormControl fullWidth>
        <LightTooltip
          title={allowOnlyOneConcept ? 'Select an Entity' : 'Select Entities'}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <FormControl>
            {allowOnlyOneConcept ? (
              <Autocomplete
                noOptionsText={noOptionsText}
                loading={loadingConcepts}
                sx={{
                  width: 'var(--concept-input-width)',
                }}
                options={possibleConceptNames}
                value={state[conditionType].form.tailValue.toString()}
                onChange={(e, value) => handleTailValueChange(e, value ?? '')}
                isOptionEqualToValue={(option, value) =>
                  option === value || value === ''
                }
                renderInput={(params) => (
                  <TextField
                    placeholder={isSubjectPrediction ? 'object' : 'subject'}
                    {...params}
                    InputProps={{
                      ...params.InputProps,
                      sx: {
                        '& input': {
                          fontSize:
                            state[conditionType].form.tailValue[0] !== ''
                              ? '0.75rem'
                              : '0.7rem',
                        },
                      },
                    }}
                    onChange={(e) =>
                      conditionType === ConditionType.hypotheses
                        ? handleConceptSearch(e.target.value)
                        : ''
                    }
                  />
                )}
              />
            ) : (
              <MultipleConceptSelector
                conditionType={conditionType}
                setFocusedInput={setFocusedInput}
                handleConceptSearch={handleConceptSearch}
                possibleConceptNames={possibleConceptNames}
                loadingConcepts={loadingConcepts}
                noOptionsText={noOptionsText}
              />
            )}
          </FormControl>
        </LightTooltip>
      </FormControl>
    </div>
  );
};

export default ConceptInputs;
