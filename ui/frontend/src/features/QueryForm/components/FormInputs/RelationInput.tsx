/*Input to choose how entities should be related*/

import './index.css';
import {
  Stack,
  FormControl,
  Select,
  MenuItem,
  Typography,
} from '@mui/material';
import LightTooltip from '../../../../shared/components/LightToolTip';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { SetStateAction } from 'react';
import {
  ConditionType,
  InputType,
} from '../../../../contexts/QueryContext/types';

interface RelationInputProps {
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  conditionType: ConditionType;
  possibleRelations: string[];
  loadingRelations: boolean;
}
const RelationInput = ({
  setFocusedInput,
  conditionType,
  possibleRelations,
  loadingRelations,
}: RelationInputProps) => {
  const { state, dispatch } = useQueryFormContext();
  return (
    <Stack
      direction="row"
      padding={0}
      sx={{
        minWidth: 'var(--min-input-width)',
        maxWidth: 'var(--max-input-width)',
      }}
    >
      <FormControl fullWidth>
        <LightTooltip
          title={`Select a Relation`}
          placement="top"
          arrow
          classes={{ tooltip: 'custom-tooltip' }}
        >
          <Select
            onMouseEnter={() => setFocusedInput(InputType.RELATION)}
            onMouseLeave={() => setFocusedInput(null)}
            MenuProps={{
              PaperProps: {
                className: 'custom-paper',
              },
            }}
            displayEmpty
            renderValue={(value: string) => {
              if (!value) {
                return (
                  <Typography
                    fontFamily={'arial-nova'}
                    fontWeight={500}
                    className="placeholder"
                    fontSize={'0.7rem'}
                    data-testid="relation-placeholder"
                  >
                    relation
                  </Typography>
                );
              }
              return <span>{value.replace(/_/, ' ')}</span>;
            }}
            labelId="relation-label"
            id="relation"
            value={state[conditionType].form.selectedRelation}
            label="relation"
            onChange={(e) =>
              dispatch({
                type: 'UPDATE_RELATION',
                formType: conditionType,
                value: e.target.value,
              })
            }
          >
            {loadingRelations ? (
              <MenuItem>{'Loading relations...'}</MenuItem>
            ) : Object.values(possibleRelations).length === 0 ? (
              <MenuItem>{'No possible relations'}</MenuItem>
            ) : (
              [
                <MenuItem key={'none'} value="">
                  --
                </MenuItem>,
                Array.from(new Set(possibleRelations)).map((relation) => (
                  <MenuItem key={relation} value={relation}>
                    {relation.replace(/_/g, ' ')}
                  </MenuItem>
                )),
              ]
            )}
          </Select>
        </LightTooltip>
      </FormControl>
    </Stack>
  );
};

export default RelationInput;
