import { mockContextualization } from './contextualization.mocks';
import {
  allDomains,
  conceptNames,
  concepts,
  filteredDomains,
  filteredEdges,
} from './data.mocks';
import {
  mockComplexityResponse,
  mockExplanation,
  mockExplanationTime,
} from './explanation.mocks';
import { constraintMock, queryMock } from './queries.mocks';
import { mockValidation } from './validation.mocks';

const mockMap: Record<string, unknown> = {
  'query/getuserqueries': [],
  'data/getedgetypes': filteredEdges,
  'data/getnodesfromdomain': concepts,
  'data/getuniquedomains': allDomains,
  'data/getnodedomains': filteredDomains,
  'data/getname': conceptNames,
  'query/': queryMock,
  'validation/': mockValidation,
  'context/contextualize': mockContextualization,
  'explain/length': mockComplexityResponse,
  'explain/': mockExplanation,
  'explain/time': mockExplanationTime,
  'query/filter_constraint': constraintMock,
};

export const getMock = (endpoint: string) => {
  const mock = mockMap[endpoint];

  if (!mock) {
    throw new Error(`No mock defined for ${endpoint}`);
  }

  return Promise.resolve(mock);
};
