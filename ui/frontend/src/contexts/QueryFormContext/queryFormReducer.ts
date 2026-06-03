/*Reducer that controls the query form state changes after certain actions */

import { PredictionType } from '../QueryContext/types';
import { ConditionType } from '../QueryContext/types';
import { AddValue } from '../QueryContext/types';
import { FormAction, SearchState, QueryFormType } from './types';
import { fixedFilters } from '../../static/filters';
import { updateSelectedFilters } from './utils';

export const optionsForVariables = ['X'];

const formInitialState: QueryFormType = {
  predictionType: PredictionType.SUBJECT,
  selectedConceptDomain: '',
  selectedRelation: '',
  tailValue: [''],
  headValue: 'X',
  addValue: AddValue.AND,
  searchType: ConditionType.hypotheses,
  headOptions: optionsForVariables,
  concepts: {},
  selectedFilters: fixedFilters,
};
export const initialState: SearchState = {
  selectedVariableDomain: '',
  [ConditionType.hypotheses]: {
    form: formInitialState,
    conditionToEdit: null,
    error: '',
  },
  [ConditionType.constraints]: {
    form: formInitialState,
    conditionToEdit: null,
    error: '',
  },
};

export const queryFormReducer = (
  state: SearchState,
  action: FormAction,
): SearchState => {
  switch (action.type) {
    case 'UPDATE_FIELD':
      return {
        ...state,
        [action.formType]: {
          ...state[action.formType],
          form: {
            ...state[action.formType].form,
            [action.field]: action.value,
          },
        },
      };
    case 'SWITCH_PREDICTION_TYPE': {
      const predictionType = action.value;
      return {
        ...state,
        [action.formType]: {
          ...state[action.formType],
          form: {
            ...state[action.formType].form,
            predictionType,
            tailValue: [''], //if the variable domain is already selected we reset the values since most likely the relation is not possible anymore
            headValue: 'X',
            selectedConceptDomain: '',
            selectedRelation: '',
          },
        },
      };
    }
    case 'UPDATE_RELATION': {
      const relation = action.value;
      return {
        ...state,
        [action.formType]: {
          ...state[action.formType],
          form: {
            ...state[action.formType].form,
            selectedRelation: relation,
          },
        },
      };
    }
    case 'UPDATE_CONCEPT_DOMAIN': {
      const updatedConceptDomain = action.value;

      return {
        ...state,
        [action.formType]: {
          ...state[action.formType],
          form: {
            ...state[action.formType].form,
            selectedConceptDomain: updatedConceptDomain,
            tailValue: [''],
          },
        },
      };
    }
    case 'UPDATE_VARIABLE_DOMAIN': {
      return {
        ...state,
        selectedVariableDomain: action.value
          .replace(/ /g, '_')
          .replace(/,/, ''),
      };
    }
    case 'SET_EDITING_CONDITION':
      return { ...state, [action.formType]: action.payload };
    case 'CLEAR':
      return {
        ...state,
        [action.formType]: {
          ...initialState[action.formType],
          form: {
            ...initialState[action.formType].form,
          },
        },
      };
    case 'RESET':
      return initialState;
    case 'SET_ERROR':
      return {
        ...state,
        [action.formType]: { ...state[action.formType], error: action.value },
      };
    case 'CANCEL_EDITING':
      return {
        ...state,
        [action.formType]: { ...state[action.formType], conditionToEdit: null },
      };
    case 'UPDATE_SELECTED_FILTERS': {
      const newSelectedFilters = updateSelectedFilters(
        fixedFilters,
        action.value,
        state[action.formType].form.selectedFilters,
      );
      return {
        ...state,
        [action.formType]: {
          ...state[action.formType],
          form: {
            ...state[action.formType].form,
            selectedFilters: newSelectedFilters,
          },
        },
      };
    }
    default:
      return state;
  }
};
