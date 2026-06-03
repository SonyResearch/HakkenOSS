/*Input to toggle between subject and object prediction*/

import './index.css';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { Stack, FormControl, Select, MenuItem } from '@mui/material';
import leftArrow from '../../../../assets/images/icons/left-long-solid.svg';
import rightArrow from '../../../../assets/images/icons/right-long-solid.svg';
import LightTooltip from '../../../../shared/components/LightToolTip';
import {
  ConditionType,
  PredictionType,
} from '../../../../contexts/QueryContext/types';

interface PredictionTypeInputProps {
  conditionType: ConditionType;
}

const PredictionTypeInput = ({ conditionType }: PredictionTypeInputProps) => {
  const { state, dispatch } = useQueryFormContext();
  return (
    <Stack sx={{ padding: 0 }}>
      <FormControl fullWidth>
        <LightTooltip title="Select a Direction" placement="top" arrow>
          <Select
            value={state[conditionType].form.predictionType}
            label="prediction-type-label"
            onChange={(e) =>
              dispatch({
                type: 'SWITCH_PREDICTION_TYPE',
                formType: conditionType,
                value: e.target.value as PredictionType,
              })
            }
          >
            <MenuItem value={PredictionType.SUBJECT}>
              <img className="direction-arrow" src={rightArrow}></img>
            </MenuItem>
            <MenuItem value={PredictionType.OBJECT}>
              <img className="direction-arrow" src={leftArrow}></img>
            </MenuItem>
          </Select>
        </LightTooltip>
      </FormControl>
    </Stack>
  );
};

export default PredictionTypeInput;
