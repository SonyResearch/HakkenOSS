import {
  AddValue,
  ConditionToEdit,
  PredictionType,
} from '../QueryContext/types';
import { ConditionType } from '../QueryContext/types';
import { Concepts } from '../../static/datasets';
import { Filters } from '../../static/filters';

export type QueryFormType = {
  predictionType: string;
  selectedConceptDomain: string;
  selectedRelation: string;
  tailValue: string[];
  headValue: string;
  addValue: AddValue;
  searchType: ConditionType;
  headOptions: string[];
  concepts: Concepts;
  selectedFilters: Filters[];
};

export interface ConditionState {
  form: QueryFormType;
  conditionToEdit: ConditionToEdit | null;
  error: string;
}

export interface SearchState {
  selectedVariableDomain: string;
  [ConditionType.hypotheses]: ConditionState;
  [ConditionType.constraints]: ConditionState;
}

export type FormAction =
  | {
      type: 'UPDATE_FIELD';
      formType: ConditionType;
      field: keyof QueryFormType;
      value: string | string[];
    }
  | { type: 'RESET' }
  | { type: 'CLEAR'; formType: ConditionType }
  | {
      type: 'UPDATE_RELATION' | 'UPDATE_CONCEPT_DOMAIN' | 'SET_ERROR';
      formType: ConditionType;
      value: string;
    }
  | {
      type: 'SWITCH_PREDICTION_TYPE';
      formType: ConditionType;
      value: PredictionType;
    }
  | {
      type: 'UPDATE_VARIABLE_DOMAIN';
      value: string;
    }
  | {
      type: 'SET_EDITING_CONDITION';
      formType: ConditionType;
      payload: ConditionState;
    }
  | { type: 'CANCEL_EDITING'; formType: ConditionType }
  | {
      type: 'UPDATE_SELECTED_FILTERS';
      formType: ConditionType;
      value: Filters;
    };

export interface FormContextType {
  state: SearchState;
  dispatch: React.Dispatch<FormAction>;
}
