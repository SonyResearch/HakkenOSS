/*Hook to access query results*/

import { useState } from 'react';

import { useQueryContext } from '../../contexts/QueryContext';
import { useQueryService } from '../../services/query';

export const useHandleQueryResults = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const { setQueryResult } = useQueryContext();

  const { getQuery: fetchQuery } = useQueryService();

  const getQuery = async (id: string) => {
    setLoading(true);

    try {
      const response = await fetchQuery(id);
      setQueryResult(response);
      setLoading(false);
      return response;
    } catch (error) {
      // TODO: Update once we have centralized error handling
      alert('Something went wrong, please try again');
      console.error(error);
      setLoading(false);
    }
  };

  return {
    getQuery,
    loading,
  };
};
