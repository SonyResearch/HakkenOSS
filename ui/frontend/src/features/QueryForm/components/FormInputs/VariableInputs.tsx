/*Input to select domain and var (currently only X) for the prediction variable*/

import './index.css';
import { FormControl, Select, MenuItem, Typography } from '@mui/material';
import LightTooltip from '../../../../shared/components/LightToolTip';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import React, { RefObject, SetStateAction } from 'react';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { optionsForVariables } from '../../../../contexts/QueryFormContext/queryFormReducer';
import {
  ConditionType,
  InputType,
  PredictionType,
} from '../../../../contexts/QueryContext/types';
import { isThereAnyVariable } from '../MainForm/utils';

interface VariableInputProps {
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  conditionType: ConditionType;
  stackRef: RefObject<HTMLDivElement>;
  possibleVariableDomains: string[];
}

const VariableInputs = ({
  setFocusedInput,
  conditionType,
  stackRef,
  possibleVariableDomains,
}: VariableInputProps) => {
  const { hypotheses, constraints } = useQueryContext();
  const { state, dispatch } = useQueryFormContext();
  const isSubjectPrediction =
    state[conditionType].form.predictionType === PredictionType.SUBJECT;
  return (
    <div className="input-wrapper variable" ref={stackRef}>
      {!isThereAnyVariable([hypotheses, constraints]) && (
        <FormControl
          sx={{
            minWidth: 'var(--min-input-width)',
            maxWidth: 'var(--max-input-width)',
          }}
        >
          <LightTooltip
            title="Select a domain"
            placement="top"
            arrow
            classes={{ tooltip: 'custom-tooltip' }}
          >
            <Select
              onMouseEnter={() => setFocusedInput(InputType.VARIABLE)}
              onMouseLeave={() => setFocusedInput(null)}
              labelId="searchTarget-label"
              id="searc-Target"
              value={state.selectedVariableDomain}
              displayEmpty
              renderValue={(value: string) => {
                if (!value) {
                  return (
                    <Typography
                      fontFamily={'arial-nova'}
                      fontWeight={500}
                      className="placeholder"
                      fontSize="0.7rem"
                    >
                      {isSubjectPrediction ? 'subject domain' : 'object domain'}
                    </Typography>
                  );
                }
                return <span>{value.replace(/_/g, ' ')}</span>;
              }}
              label="domain"
              onChange={(e) => {
                dispatch({
                  type: 'UPDATE_VARIABLE_DOMAIN',
                  value: e.target.value,
                });
              }}
            >
              {possibleVariableDomains.length === 0 ? (
                <MenuItem>{'No possible domains'}</MenuItem>
              ) : (
                [
                  <MenuItem key={'none'} value="">
                    --
                  </MenuItem>,
                  possibleVariableDomains.map((varDomain, index) => (
                    <MenuItem key={index} value={varDomain}>
                      {varDomain}
                    </MenuItem>
                  )),
                ]
              )}
            </Select>
          </LightTooltip>
        </FormControl>
      )}
      <FormControl style={{ minWidth: '60px' }}>
        <LightTooltip
          title={'Select a Variable'}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <Select
            onMouseEnter={() => setFocusedInput(InputType.VARIABLE)}
            onMouseLeave={() => setFocusedInput(null)}
            value={state[conditionType].form.headValue}
            displayEmpty
            disabled
            renderValue={(value: string) => {
              if (!value) {
                return (
                  <Typography
                    fontFamily={'arial-nova'}
                    fontWeight={500}
                    className="placeholder"
                    fontSize="0.7rem"
                  >
                    X
                  </Typography>
                );
              }
              return <span>{value}</span>;
            }}
            onChange={(e) => {
              dispatch({
                type: 'UPDATE_FIELD',
                field: 'headValue',
                formType: conditionType,
                value: e.target.value,
              });
            }}
          >
            {optionsForVariables.map((option) => (
              <MenuItem key={option} value={option}>
                {option}
              </MenuItem>
            ))}
          </Select>
        </LightTooltip>
      </FormControl>
    </div>
  );
};

export default VariableInputs;
