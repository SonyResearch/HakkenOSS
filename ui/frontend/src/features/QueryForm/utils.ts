import {
  createNewCondition,
  getOptionsFromConcepts,
  updateConcepts,
} from './components/MainForm/utils';
import { searchConcepts } from './components/MainForm/utils';
import { domains } from '../../static/datasets';
import { Domain } from '../../static/datasets/data-types';
import { FormAction, SearchState } from '../../contexts/QueryFormContext/types';
import React, { Dispatch, SetStateAction } from 'react';
import {
  Condition,
  ConditionType,
  PredictionType,
} from '../../contexts/QueryContext/types';

export const createDebouncedSearch = <T>(
  delay = 500,
  domain: string,
  setConcepts: React.Dispatch<SetStateAction<T[]>>,
  setLoading: React.Dispatch<SetStateAction<boolean>>,
  onlyConceptName: boolean,
) => {
  let timeout: ReturnType<typeof setTimeout>;
  return (query: string) => {
    setLoading(true);
    clearTimeout(timeout);
    timeout = setTimeout(async () => {
      try {
        const updatedConcepts =
          query.length > 1
            ? await searchConcepts(domain, query)
            : await updateConcepts(domain);
        const result = onlyConceptName
          ? getOptionsFromConcepts(updatedConcepts)
          : updatedConcepts;

        setConcepts(result as T[]);
      } catch (error) {
        console.error(error);
        (error as Error).message ??
          'Something went wrong, please try again later';
      } finally {
        setLoading(false);
      }
    }, delay);
  };
};

export const editAndShiftConditions = (
  conditions: Record<number, Condition>,
  newConditions: Record<number, Condition>,
) => {
  const allConditions: Record<number, Condition> = {};
  const conditionKeys = Object.keys(conditions);
  const newConditionKeys = Object.keys(newConditions);
  const editingIndex = Number(newConditionKeys[0]);

  let currentIndex = 0;
  for (let i = 0; i < conditionKeys.length; i++) {
    if (Number(conditionKeys[i]) != editingIndex) {
      allConditions[currentIndex] = conditions[Number(conditionKeys[i])];
      currentIndex++;
    } else {
      for (let j = 0; j < newConditionKeys.length; j++) {
        allConditions[currentIndex] =
          newConditions[Number(newConditionKeys[j])];
        currentIndex++;
      }
    }
  }

  return allConditions;
};

export const addCondition = async (
  state: SearchState,
  dispatch: Dispatch<FormAction>,
  checkExistingVariable: (arg1: string, arg2: Domain) => string,
  handleAddNewVariable: (arg1: string, arg2: Domain) => void,
  handleAddNewConditions: (
    arg1: Record<number, Condition>,
    arg2: ConditionType,
  ) => void,
  nextIndex: number,
  conditionType: ConditionType,
) => {
  const newConditions: Record<number, Condition> = {};
  const variableDomain: Domain = domains[state.selectedVariableDomain];
  const tailValue = state[conditionType].form.tailValue;
  const conceptNames = Array.isArray(tailValue) ? tailValue : [tailValue];

  if (!conceptNames.length)
    throw new Error(
      'At least one concept must be selected to create a condition',
    );

  const formConditions = await Promise.all(
    conceptNames.map(async (conceptName) => {
      try {
        return await createNewCondition(
          state.selectedVariableDomain,
          state[conditionType].form.selectedRelation,
          state[conditionType].form.selectedConceptDomain,
          conceptName,
          state[conditionType].form.predictionType,
        );
      } catch (error) {
        console.error(`Error creating condition for ${conceptName}:`, error);
        throw new Error((error as Error).message);
      }
    }),
  );

  formConditions.forEach((condition, index) => {
    const varExists = checkExistingVariable('X', variableDomain);
    if (varExists !== 'exists') {
      handleAddNewVariable('X', variableDomain);
    }

    if (state[conditionType].form.predictionType === PredictionType.OBJECT) {
      const temp = condition.tail;
      condition.tail = condition.head;
      condition.head = temp;
    }
    const conditionIndex: number = state[conditionType].conditionToEdit
      ? (state[conditionType].conditionToEdit?.index ?? 0) + index + index
      : nextIndex + index;
    newConditions[conditionIndex] = {
      condition: condition,
      conditionType,
      addValue: state[conditionType].form.addValue,
    };
  });
  if (Object.values(newConditions).length) {
    handleAddNewConditions(newConditions, conditionType);
    dispatch({ formType: conditionType, type: 'CLEAR' });
  }
};

export const checkIfOnlyOneCondition = (
  conditionLengths: Record<ConditionType, number>,
) => {
  let total = 0;
  for (const length of Object.values(conditionLengths)) {
    total += length;
    if (total > 1) return false;
  }
  return total === 1;
};

export const getNextIndex = (conditions: Record<number, Condition>) => {
  const conditionKeys = Object.keys(conditions);
  const newIndex = conditionKeys.length
    ? Number(conditionKeys[conditionKeys.length - 1]) + 1
    : 0;
  return newIndex;
};
