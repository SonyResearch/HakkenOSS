import { defaultConfig } from '../../configI';
import { AddValue, Condition, ConditionType, QueryMode } from './types';

export const defaultResult = {
  candidates: [],
  createdAt: '',
  id: '',
  query: '',
  queryId: '',
  updatedAt: '',
  userId: '',
};

export const defaultCondition: Condition = {
  addValue: AddValue.AND,
  conditionType: ConditionType.hypotheses,
  condition: {
    head: {
      domain: '',
      label: '',
      id: '',
      isVariable: true,
    },
    relation: '',
    tail: {
      domain: '',
      label: '',
      id: '',
      isVariable: false,
    },
  },
};

export const defaultData = {
  hypotheses: {},
  constraints: {},
  variables: [],
  query: '',
  queryApi: '',
  queryResult: defaultResult,
  candidatesResult: { candidates: [] },
  example: 0,
  candidatesNumber: defaultConfig.candidatesNumber,
  searchedParameters: undefined,
  queryMode: 'simple' as QueryMode,
  isSearching: false,
  setHypotheses: () => undefined,
  setConstraints: () => undefined,
  setQuery: () => undefined,
  setQueryApi: () => undefined,
  setQueryResult: () => undefined,
  setVariables: () => undefined,
  resetToDefault: () => undefined,
  setCandidatesResult: () => undefined,
  setExample: () => undefined,
  setCandidatesNumber: () => undefined,
  setSearchedParameters: () => undefined,
  setQueryMode: () => undefined,
  setIsSearching: () => undefined,
};

export const CONSTANTS = {
  AND: 'AND',
  AND_NOT: 'AND NOT',
  BELONGS_TO: 'BELONGS_TO',
  OR: 'OR',
};
