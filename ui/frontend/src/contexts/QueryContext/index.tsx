/* Context storing all states related to query for easy access across the components that need it */

import { createContext, useContext, useState } from 'react';
import { CandidatesResultType } from '../../features/CandidateDashboard/types';

import {
  Condition,
  IQueryContext,
  Props,
  QueryResult,
  Variable,
  SearchedParameters,
  QueryMode,
} from './types';
import { defaultData, defaultResult } from './utils';
import { defaultConfig } from '../../configI';

export const QueryContext = createContext<IQueryContext>(defaultData);

export const QueryProvider = ({ children }: Props): JSX.Element => {
  const [queryMode, setQueryMode] = useState<QueryMode>('simple');
  const [query, setQuery] = useState<string>(defaultData.query);
  const [queryApi, setQueryApi] = useState<string>('');
  const [variables, setVariables] = useState<Record<number, Variable>>({});
  const [hypotheses, setHypotheses] = useState<Record<number, Condition>>({});
  const [constraints, setConstraints] = useState<Record<number, Condition>>({});
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [example, setExample] = useState<number | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResult>(defaultResult);
  const [candidatesResult, setCandidatesResult] =
    useState<CandidatesResultType>({ candidates: [] });
  const [candidatesNumber, setCandidatesNumber] = useState<number>(
    defaultConfig.candidatesNumber,
  );
  const [searchedParameters, setSearchedParameters] = useState<
    SearchedParameters | undefined
  >(undefined);

  const resetToDefault = () => {
    setQuery(defaultData.query);
    setQueryApi(defaultData.queryApi);
    setVariables(defaultData.variables);
    setHypotheses(defaultData.hypotheses);
    setConstraints(defaultData.constraints);
    setQueryResult(defaultData.queryResult);
    setIsSearching(false);
    setExample(null);
  };

  return (
    <QueryContext.Provider
      value={{
        hypotheses,
        constraints,
        query,
        queryApi,
        queryResult,
        variables,
        candidatesResult,
        example,
        candidatesNumber,
        searchedParameters,
        queryMode,
        isSearching,
        setHypotheses,
        setConstraints,
        setVariables,
        setQuery,
        setQueryApi,
        setQueryResult,
        resetToDefault,
        setCandidatesResult,
        setExample,
        setCandidatesNumber,
        setSearchedParameters,
        setQueryMode,
        setIsSearching,
      }}
    >
      {children}
    </QueryContext.Provider>
  );
};

export const useQueryContext = (): IQueryContext => useContext(QueryContext);
