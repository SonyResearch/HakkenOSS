/*Search filter selection*/

import './index.css';
import {
  Box,
  Checkbox,
  Chip,
  FormControl,
  ListItemText,
  Menu,
  MenuItem,
} from '@mui/material';
import LightTooltip from '../../../../shared/components/LightToolTip';
import React, { RefObject, SetStateAction, useRef, useState } from 'react';
import { Filters, fixedFilters } from '../../../../static/filters';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import {
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';

interface FilterInputProps {
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  conditionType: ConditionType;
  stackRef: RefObject<HTMLDivElement>;
}

const FilterInput = ({
  setFocusedInput,
  conditionType,
  stackRef,
}: FilterInputProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState<boolean>(false);
  const inputRef = useRef();
  const { state, dispatch } = useQueryFormContext();

  return (
    <FormControl fullWidth sx={{ width: '9rem' }}>
      <LightTooltip
        title={`Select a Filter`}
        placement="top"
        arrow
        classes={{ tooltip: 'custom-tooltip' }}
      >
        <Box
          ref={stackRef}
          onMouseLeave={() => setFocusedInput(null)}
          //onMouseOver={() => setFocusedInput(InputType.FILTER)}
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <Box ref={inputRef} className="filter-box">
            {state[conditionType].form.selectedFilters.map((filter) => (
              <Chip
                key={filter}
                label={filter}
                sx={{
                  fontSize: '0.7rem',
                  backgroundColor: 'rgb(242, 247, 255)',
                  marginRight: '2rem',
                }}
                className="filter-tag"
              ></Chip>
            ))}
          </Box>
          <Menu anchorEl={inputRef.current} open={isMenuOpen}>
            {Object.values(Filters).map((filter) => (
              <MenuItem
                disabled={fixedFilters.includes(filter)}
                key={filter}
                onClick={() =>
                  dispatch({
                    type: 'UPDATE_SELECTED_FILTERS',
                    formType: conditionType,
                    value: filter,
                  })
                }
              >
                <Checkbox
                  checked={state[conditionType].form.selectedFilters.includes(
                    filter,
                  )}
                />
                <ListItemText primary={filter} />
              </MenuItem>
            ))}
          </Menu>
        </Box>
      </LightTooltip>
    </FormControl>
  );
};

export default FilterInput;
