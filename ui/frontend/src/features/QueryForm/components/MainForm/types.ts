import {
  Condition,
  QueryResult,
} from '../../../../contexts/QueryContext/types';

export interface QueryHistoryItem extends QueryResult {
  hypotheses: Record<number, Condition>;
  constraints: Record<number, Condition>;
  queryString: string;
}
