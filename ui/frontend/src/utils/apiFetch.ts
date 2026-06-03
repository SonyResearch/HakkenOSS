// api/client.ts
import { appConfig } from '../config';
import { getMock } from '../mocks/getMock';

export async function fetchGateway(
  endpoint: string,
  method: 'GET' | 'POST' | 'DELETE',
  payload: unknown,
  options: RequestInit = {},
  query: Record<string, string | number | boolean | null | undefined>,
) {
  const baseUrl = appConfig.apiGatewayUrl.replace(/\/$/, '');
  const cleanEndpoint = endpoint.replace(/^\//, '');

  let url = `${baseUrl}/${cleanEndpoint}`;

  if (query) {
    const queryString = Object.entries(query)
      .filter(([, value]) => value !== null && value !== undefined)
      .map(
        ([key, value]) =>
          `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`,
      )
      .join('&');

    if (queryString) {
      url += `?${queryString}`;
    }
  }

  if (appConfig.queryDemo) {
    return getMock(endpoint);
  }

  const response = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...(payload !== undefined && method !== 'GET'
      ? { body: JSON.stringify(payload) }
      : {}),
    ...options,
  });

  if (!response.ok) {
    throw new Error(`API error ${response.status} for ${endpoint}`);
  }

  return response.json();
}
