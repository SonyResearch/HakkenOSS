import { appConfig } from '../../config';
import { candidateDTO2Candidate, snakeToCamel } from '../converters';
import {
  Condition,
  QueryMode,
  QueryResult,
} from '../../contexts/QueryContext/types';
import { fetchGateway } from '../../utils/apiFetch';

const CONSTANTS = {
  QUERIES: 'queries',
  QUERY: 'query',
};
const apiGatewayUrl = appConfig.apiGatewayUrl;

export const useQueryService = () => {
  const submitQuery = async (
    queryApi: string,
    queryString: string,
    hypotheses: Record<number, Condition>,
    constraints: Record<number, Condition>,
    candidatesNumber: number,
    queryMode: QueryMode,
  ) => {
    const parsedQuery = JSON.parse(queryApi);
    const payload = {
      queryApi: parsedQuery,
      queryString,
      hypotheses,
      constraints,
      candidatesNumber,
      queryMode,
    };
    try {
      const result = await fetchGateway(
        `${CONSTANTS.QUERY}/`,
        'POST',
        payload,
        {},
        {},
      );
      const candidates = result.candidates.map(candidateDTO2Candidate);

      if (!candidates.length) {
        throw new Error('No results found for this query');
      }
      return { candidates };
    } catch (error) {
      console.error('Something went wrong processing this query: ', error);
      throw new Error('Something went wrong processing this query');
    }
  };

  const getQuery = async (id: string) => {
    const query = { id };
    const response: QueryResult = await fetchGateway(
      ` ${CONSTANTS.QUERY}/getquery`,
      'GET',
      {},
      {},
      query,
    ).then((result) => result);

    return snakeToCamel(response);
  };

  const getUserQueries = async () => {
    try {
      const response = await fetchGateway(
        `${CONSTANTS.QUERY}/getuserqueries`,
        'GET',
        {},
        {},
        {},
      );
      const queries = response.queries;
      const parsedQueries = snakeToCamel(queries);
      return parsedQueries;
    } catch (error) {
      console.error(error);
      throw new Error('Something went wrong getting user queries');
    }
  };

  const deleteUserQuery = async (queryId: string) => {
    const query = { id: queryId };
    try {
      await fetchGateway(
        `${apiGatewayUrl}${CONSTANTS.QUERY}/deletequery`,
        'DELETE',
        {},
        {},
        query,
      );
      const updatedQueries = await getUserQueries();
      return snakeToCamel(updatedQueries);
    } catch (error) {
      console.error(error);
      throw new Error(
        'Something went wrong deleting user query with id: ' + queryId,
      );
    }
  };

  const clearUserQueries = async () => {
    try {
      await fetchGateway(
        `${apiGatewayUrl}${CONSTANTS.QUERY}/deleteuserqueries`,
        'DELETE',
        {},
        {},
        {},
      );
    } catch (error) {
      console.error(error);
      throw new Error('Something went wrong deleting user queries');
    }
  };

  return {
    submitQuery,
    getQuery,
    getUserQueries,
    deleteUserQuery,
    clearUserQueries,
  };
};
