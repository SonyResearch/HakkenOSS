import { Concept } from '../../static/datasets';

export interface VarAssignments {
  [key: string]: string;
}
export interface ConditionsScore {
  [key: string]: number;
}

export interface CandidatesResultType {
  candidates: CandidateResultType[];
  concepts?: { [s: string]: Concept };
}

export interface CandidateResultType {
  variableAssignments: VarAssignments;
  queryScore: number;
  conditionsScores: ConditionsScore;
  name?: string;
  domain?: string;
}

export interface PublicationInfo {
  publication_id: string;
  year: number;
  title: string;
  doi?: string;
  pmid?: string;
  pmcid?: string;
  authors: { first_name: string; last_name: string }[];
  abstract?: string;
  citations_count: number | 'None';
}

export interface Reference {
  publication_info: PublicationInfo;
  score: number;
  text: string;
  summary: string;
}

export interface Source {
  id: string;
  eissn: string;
  issn: string;
  original_title: string;
  title: string;
}

export interface ContextualizationResult {
  status?: number;
  references: Reference[];
  summary: string;
}

export type SortingCategories = 'score' | 'citations_count' | 'year';

export type Sorting = {
  category: SortingCategories;
  order: 'ascending' | 'descending';
};

export type Filters = {
  author: string;
  title: string;
  abstract: string;
};

export type ResultViews = 'context' | 'explanation';
