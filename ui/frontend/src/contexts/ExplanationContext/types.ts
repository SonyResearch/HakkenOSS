import { Triple } from '../QueryContext/types';

export interface ParsedExplanation {
  predictedTriple: Triple;
  explanations: ParsedExplanationItem[];
  queryId: string;
}

export interface ParsedExplanationItem {
  data: Triple[];
  length: number;
  score: number;
}

export interface RawExplanationResult {
  explanations: Record<string, RawExplanationItem[]>;
}

export interface RawExplanationItem {
  data: string;
  length: number;
  score: number;
}

export interface ExplanationComplexityResponse {
  length_dict: Record<string, number>;
}
