import { loadEnv } from 'vite';
import svgr from 'vite-plugin-svgr';
import viteTsconfigPaths from 'vite-tsconfig-paths';
import { defineConfig } from 'vitest/config';

import react from '@vitejs/plugin-react';

import { defaultConfig } from './src/configI';

// https://vitejs.dev/config/

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  const PORT = Number(env.VITE_PORT) || 3000;
  const appConfig = {
    ...defaultConfig,
    ...Object.fromEntries(
      Object.entries({
        apiPath: env.VITE_API_PATH,
        apiGatewayUrl: env.VITE_API_GATEWAY_URL,
        apiUrl: env.VITE_API_URL,
      }).filter(([_, value]) => value !== undefined),
    ),
  };

  return {
    server: {
      host: true,
      port: PORT,
      proxy: {
        [appConfig.apiPath]: {
          target: appConfig.apiUrl,
          //         changeOrigin: true,  // Needed to avoid CORS issues
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              const auth = req.headers['Authorization'] || env.VITE_AUTH_JWT;
              if (auth) {
                proxyReq.setHeader('Authorization', auth);
              }
            });
          },
          rewrite: (path) =>
            path.replace(new RegExp(`^${appConfig.apiPath}`), ''),
        },
        '/api-gateway': {
          target: appConfig.apiGatewayUrl,
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              const auth = req.headers['Authorization'] || env.VITE_AUTH_JWT;
              if (auth) {
                proxyReq.setHeader('Authorization', auth);
              }
            });
          },
          rewrite: (path) => path.replace(new RegExp(`^/api-gateway`), ''),
        },
      },
    },
    plugins: [
      react(),
      viteTsconfigPaths(),
      svgr({
        include: '**/*.svg?react',
      }),
    ],
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: './vitest.setup.ts',
    },
  };
});
