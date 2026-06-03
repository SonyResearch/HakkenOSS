/*Toggle to choose between complex and simple query usage*/

import React, { SetStateAction } from 'react';
import './index.css';
import {
  Condition,
  ConditionType,
} from '../../../../contexts/QueryContext/types';
import LightTooltip from '../../../../shared/components/LightToolTip';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { defaultData } from '../../../../contexts/QueryContext/utils';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';

interface QueryModeToggleProps {
  tempConstraints: Record<number, Condition>;
  setTempConstraints: React.Dispatch<SetStateAction<Record<number, Condition>>>;
}

export const QueryModeToggle = ({
  tempConstraints,
  setTempConstraints,
}: QueryModeToggleProps) => {
  const {
    queryMode,
    setQueryMode,
    setHypotheses,
    hypotheses,
    constraints,
    setConstraints,
  } = useQueryContext();
  const { state, dispatch } = useQueryFormContext();

  const queryModeDescription = {
    simple:
      'Explore candidates for a single relationship. You can add constraints that all potential candidates must meet',
    complex: 'Explore candidates that satisfy a combination of relationships.',
  };

  const handleQueryModeToggle = () => {
    if (queryMode === 'complex') {
      setHypotheses(Object.fromEntries(Object.entries(hypotheses).slice(0, 1)));
      setConstraints(tempConstraints);
    } else {
      setConstraints(defaultData.constraints);
      setTempConstraints(constraints);
    }
    if (state[ConditionType.hypotheses].conditionToEdit) {
      dispatch({ type: 'CANCEL_EDITING', formType: ConditionType.hypotheses });
    }
    if (state[ConditionType.constraints].conditionToEdit) {
      dispatch({ type: 'CANCEL_EDITING', formType: ConditionType.constraints });
    }
    setQueryMode(queryMode === 'simple' ? 'complex' : 'simple');
  };
  return (
    <nav className="mode-toggle">
      <LightTooltip
        title={queryModeDescription['simple']}
        placement="top"
        arrow
      >
        <button
          className={queryMode === 'simple' ? 'active' : ''}
          onClick={handleQueryModeToggle}
        >
          Single Shot
        </button>
      </LightTooltip>
      <LightTooltip
        title={queryModeDescription['complex']}
        placement="top"
        arrow
      >
        <button
          className={queryMode === 'complex' ? 'active' : ''}
          onClick={handleQueryModeToggle}
        >
          Combo
        </button>
      </LightTooltip>
    </nav>
  );
};
