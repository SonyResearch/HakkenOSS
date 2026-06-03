import { Domain, Relation } from '../../static/datasets/data-types';
import { CandidatesResultType } from '../../features/CandidateDashboard/types';

export interface Props {
  children: React.ReactNode;
}

export interface HeadTail extends Relation {
  label: 'Variable' | 'Concept';
}

export interface Query {
  formula: string;
  variables: Variable[];
}

export interface SearchedParameters {
  query: string;
  hypotheses: Record<number, Condition>;
  constraints: Record<number, Condition>;
  candidatesNumber: number;
}

export type QueryMode = 'simple' | 'complex';

export interface IQueryContext {
  hypotheses: Record<number, Condition>;
  constraints: Record<number, Condition>;
  query: string;
  queryApi: string;
  queryResult: QueryResult;
  variables: Record<number, Variable>;
  candidatesResult: CandidatesResultType;
  example: number | null;
  candidatesNumber: number;
  searchedParameters: SearchedParameters | undefined;
  queryMode: QueryMode;
  isSearching: boolean;
  setHypotheses: (hypotheses: Record<number, Condition>) => void;
  setConstraints: (constraints: Record<number, Condition>) => void;
  setQuery: (query: string) => void;
  setQueryApi: (queryApi: string) => void;
  setQueryResult: (data: QueryResult) => void;
  setVariables: (variables: Record<number, Variable>) => void;
  resetToDefault: () => void;
  setCandidatesResult: (candidatesResult: CandidatesResultType) => void;
  setExample: (example: number | null) => void;
  setCandidatesNumber: (candidates: number) => void;
  setSearchedParameters: (
    searchParameters: SearchedParameters | undefined,
  ) => void;
  setQueryMode: (queryMode: QueryMode) => void;
  setIsSearching: (isSearching: boolean) => void;
}

export interface TripleEntity {
  isVariable: boolean;
  label: string;
  domain: string;
  id: string;
}

export interface ConditionItem {
  head: TripleEntity;
  relation: string;
  tail: TripleEntity;
}

export interface Condition {
  condition: ConditionItem;
  conditionType: ConditionType;
  addValue: AddValue; // Added by nagano
}

export interface ConditionToEdit {
  condition: Condition;
  index: number;
}

export interface Variable {
  label: string;
  domain: Domain;
}
export interface QueryResult {
  candidates: Candidate[];
  createdAt: string;
  queryId: string;
  id: string;
  query: string;
  updatedAt: string;
  userId: string;
}

export interface CandidateDTO {
  var_assignments: {
    [key: string]: string;
  }[];
  query_score: number;
  condition_scores: {
    [key: string]: number;
  }[];
}

export interface ParsedCandidate {
  variableAssignments: {
    [key: string]: string;
  }[];
  queryScore: number;
  conditionsScores: {
    [key: string]: number;
  }[];
}

export interface Candidate {
  conditionsScores: ConditionsScores[];
  id: string;
  queryScore: number;
  variableAssignments: VariableAssignments[];
}

export interface VariableAssignments {
  concept: string;
  id: string;
  variable: string;
}

export interface ConditionsScores {
  concept: string;
  domain: string;
  head: string;
  id: string;
  relation: string;
  score: number | boolean;
  type: ConditionType;
  tail: string;
}

export enum ConditionType {
  //IS_A = 'isA',  Not used probably ever
  //IS_NOVEL = 'isNovel',  //Not used in MVP0.1
  hypotheses = 'P',
  constraints = 'EXISTS',
}

export enum AddValue {
  AND = 'AND',
  AND_NOT = 'AND NOT',
  OR = 'OR',
}

export enum PredictionType {
  SUBJECT = 'subject',
  OBJECT = 'object',
}

export enum InputType {
  VARIABLE = 'variable',
  RELATION = 'relation',
  CONCEPT = 'concept',
  //FILTER = 'filter',
}

export interface TripleContext {
  subject: string;
  relation: string;
  object: string;
}

export interface Triple {
  head: string;
  relation: string;
  tail: string;
}

export interface Clause {
  triples: Triple[];
  matchPattern: string;
  relevance: number;
  message: string;
}

// export interface QueryResponseType {
//   message: string;
//   data: any;
//   responseData: string;
//   status?: number;
// }
