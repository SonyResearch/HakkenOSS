import { DataSets } from './static/datasets';

export interface AppConfig {
  apiPath: string;
  apiUrl: string;
  apiGatewayUrl: string;
  appTitle: string;
  candidatesNumber?: number;
  conceptLimit?: number;
  timeout?: number;
  dataSet?: DataSets;
  queryDemo?: boolean;
}

export const defaultConfig = {
  apiPath: '/api/',
  apiUrl: 'http://localhost:8080/',
  apiGatewayUrl: '/api-gateway',
  appTitle: 'Hakken UI',
  candidatesNumber: 8,
  conceptLimit: 300,
  timeout: 300,
  dataSet: 'pubtator' as DataSets,
  queryDemo: false,
};
