/*This context is used to access the state from ./queryFormReducer.ts across all components that need to access any of the form parameters*/

import React, { createContext, useReducer, useContext } from 'react';
import { FormContextType } from './types';
import { queryFormReducer, initialState } from './queryFormReducer';

const QueryFormContext = createContext<FormContextType | null>(null);

interface QueryFormProviderProps {
  children: React.ReactNode;
}

export const QueryFormProvider = ({ children }: QueryFormProviderProps) => {
  const [state, dispatch] = useReducer(queryFormReducer, initialState);

  return (
    <QueryFormContext.Provider value={{ state, dispatch }}>
      {children}
    </QueryFormContext.Provider>
  );
};

export const useQueryFormContext = (): FormContextType => {
  const context = useContext(QueryFormContext);
  if (!context) {
    throw new Error('useQueryFormContext has to be used with a provider');
  }
  return context;
};
