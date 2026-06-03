/*Operator selection OR/AND/AND NOT input*/

import './index.css';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { Stack, FormControl, MenuItem, Select } from '@mui/material';
import LightTooltip from '../../../../shared/components/LightToolTip';
import {
  AddValue,
  ConditionType,
} from '../../../../contexts/QueryContext/types';

interface BoolAndTypeInputProps {
  conditionType: ConditionType;
  isFirstConditionOfType: boolean;
}

const BoolAndTypeInput = ({
  conditionType,
  isFirstConditionOfType,
}: BoolAndTypeInputProps) => {
  const { state, dispatch } = useQueryFormContext();
  return (
    <Stack padding={0}>
      <FormControl>
        <LightTooltip title="Select an Operator" placement="top" arrow>
          <Select
            fullWidth
            labelId="bool-operation-label"
            id="bool-operation"
            label="Boolean Operation"
            value={state[conditionType].form.addValue}
            onChange={(e) =>
              dispatch({
                type: 'UPDATE_FIELD',
                formType: conditionType,
                field: 'addValue',
                value: e.target.value,
              })
            }
          >
            <MenuItem value={AddValue.AND}>
              {isFirstConditionOfType ? '--' : 'AND'}
            </MenuItem>
            {
              <MenuItem value={AddValue.AND_NOT}>
                {isFirstConditionOfType ? 'NOT' : 'AND NOT'}
              </MenuItem>
            }
            {!isFirstConditionOfType && (
              <MenuItem value={AddValue.OR}>OR</MenuItem>
            )}
          </Select>
        </LightTooltip>
      </FormControl>
      {/* TODO: Update when we're able to implement filters
        <FormControl>
          <InputLabel shrink={true} id="search-type-label">
            Type
          </InputLabel>
          <LightTooltip title="Select a Search Type" placement="top" arrow>
            <Select
              sx={{ width: { md: '100px', lg: 'min(8vw, 150px)' } }}
              labelId="search-type-label"
              id="search-type"
              label="Type"
              value={state.form.searchType}
              onChange={(e) =>
                dispatch({
                  type: 'UPDATE_FIELD',
                  field: 'searchType',
                  value: e.target.value,
                })
              }
            >
              {Object.entries(ConditionType)?.map(([k, v]) => (
                <MenuItem key={v} value={v}>
                  {k}
                </MenuItem>
              ))}
            </Select>
          </LightTooltip>
        </FormControl>
      )}*/}
    </Stack>
  );
};

export default BoolAndTypeInput;
