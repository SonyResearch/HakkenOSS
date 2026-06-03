import { AppConfig, defaultConfig } from './configI';

const envConfig: AppConfig = {
  apiPath: import.meta.env.VITE_API_PATH,
  apiUrl: import.meta.env.VITE_API_URL,
  apiGatewayUrl: import.meta.env.VITE_API_GATEWAY_URL,
  appTitle: import.meta.env.VITE_APP_TITLE,
};
export const appConfig: AppConfig = {
  ...defaultConfig,
  ...Object.fromEntries(
    Object.entries(envConfig).filter(([, value]) => value !== undefined),
  ),
};
