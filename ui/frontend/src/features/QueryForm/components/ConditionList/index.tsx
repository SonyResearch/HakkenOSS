import './index.css';
import { useQueryContext } from '../../../../contexts/QueryContext';
import plusIcon from '../../../../assets/images/icons/plus-solid.svg';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import ConditionRow from '../ConditionRow';
import { defaultCondition } from '../../../../contexts/QueryContext/utils';
import { getTripleFromCondition } from '../MainForm/utils';
import { Conditions } from '../../../../static/useCases';
import {
  Condition,
  ConditionType,
} from '../../../../contexts/QueryContext/types';
import React, { SetStateAction, useEffect } from 'react';
import { InputType } from '../../../../contexts/QueryContext/types';
import { PyramidToggle } from '../Pyramids';
import { ThemeProvider } from '@emotion/react';
import { conditionListTheme } from '../../themes';
import { checkIfOnlyOneCondition, getNextIndex } from '../../utils';

interface ConditionListProps {
  conditionType: ConditionType;
  conditions: Conditions;
  setConditions: (prevConditions: Record<number, Condition>) => void;
  focusedInput: InputType | null;
  setFocusedInput: React.Dispatch<SetStateAction<InputType | null>>;
  isGuideVisible: boolean;
  setIsGuideVisible: React.Dispatch<SetStateAction<boolean>>;
  conditionLengths: Record<ConditionType, number>;
}

const ConditionList = ({
  focusedInput,
  setFocusedInput,
  isGuideVisible,
  setIsGuideVisible,
  conditionType,
  conditions,
  setConditions,
  conditionLengths,
}: ConditionListProps) => {
  const { state, dispatch } = useQueryFormContext();
  const { hypotheses, queryMode } = useQueryContext();
  const conditionListTitle =
    conditionType === ConditionType.constraints
      ? 'CONSTRAINTS'
      : queryMode === 'simple'
        ? 'HYPOTHESIS'
        : 'HYPOTHESES';

  const shouldToggleBeVisible =
    conditionType === ConditionType.hypotheses &&
    state[ConditionType.hypotheses].conditionToEdit;

  const firstHypothesis = Object.values(hypotheses)[0];

  const canAddMoreConditions =
    queryMode === 'complex' || conditionType === ConditionType.constraints;

  useEffect(() => {
    if (!conditionLengths[ConditionType.hypotheses]) {
      hypotheses[0] = defaultCondition;
    }
    if (
      Object.values(hypotheses)[0] &&
      !Object.values(hypotheses)[0].condition.head.domain &&
      conditionType === ConditionType.hypotheses
    ) {
      dispatch({
        type: 'SET_EDITING_CONDITION',
        formType: ConditionType.hypotheses,
        payload: {
          ...state[ConditionType.hypotheses],
          conditionToEdit: {
            condition: defaultCondition,
            index: 0,
          },
        },
      });
    }
  }, [hypotheses, queryMode]);

  const handleAddButton = () => {
    //refactor
    const { variable } = getTripleFromCondition(firstHypothesis);
    if (
      state[ConditionType.hypotheses].conditionToEdit ||
      state[ConditionType.constraints].conditionToEdit
    ) {
      dispatch({
        type: 'SET_ERROR',
        formType: conditionType,
        value: 'Finish editing a condition before adding a new one',
      });
    } else {
      const newIndex = getNextIndex(conditions);
      const newCondition = {
        ...defaultCondition,
        condition: {
          ...defaultCondition.condition,
          head: variable,
        },
      };
      setConditions({ ...conditions, [newIndex]: newCondition });
      dispatch({
        type: 'SET_EDITING_CONDITION',
        formType: conditionType,
        payload: {
          ...state[conditionType],
          conditionToEdit: {
            condition: conditions[newIndex],
            index: newIndex,
          },
        },
      });
    }
  };

  return (
    <ThemeProvider theme={conditionListTheme}>
      <h4 className="condition-list-title">{conditionListTitle}</h4>
      <div className={`conditions-container`}>
        {shouldToggleBeVisible && (
          <PyramidToggle
            isGuideVisible={isGuideVisible}
            setIsGuideVisible={setIsGuideVisible}
            highlight={focusedInput}
          />
        )}
        {state[conditionType] &&
        conditionLengths[conditionType] > 0 &&
        state[conditionType].form ? (
          Object.entries(conditions)?.map(([key, value], index) => {
            return (
              <ConditionRow
                key={key}
                condition={value}
                listing={index}
                index={Number(key)}
                conditions={conditions}
                setConditions={setConditions}
                setFocusedInput={setFocusedInput}
                conditionType={conditionType}
                isOnlyCondition={checkIfOnlyOneCondition(conditionLengths)}
                isGuideVisible={isGuideVisible}
                focusedInput={focusedInput}
                canAddMoreConditions={canAddMoreConditions}
              ></ConditionRow>
            );
          })
        ) : (
          <p className="empty-conditions-text">
            There are no {conditionListTitle.toLocaleLowerCase()} yet
          </p>
        )}
        {canAddMoreConditions && (
          <img
            src={plusIcon}
            alt="icon to add conditions"
            onClick={handleAddButton}
          ></img>
        )}
      </div>
      {state[conditionType].error && (
        <p className="error-message">{state[conditionType].error}</p>
      )}
    </ThemeProvider>
  );
};
export default ConditionList;
