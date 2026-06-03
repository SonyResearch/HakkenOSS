# UI

The Hakken user interface — a React/TypeScript frontend that communicates with the Hakken API gateway to access the underlying microservices.

## Structure

```
ui/
├── frontend/   # React/TypeScript application (Vite)
└── backend/    # FastAPI gateway (bridges the frontend to Kubernetes services)
```

## Frontend

See [`frontend/README.md`](frontend/README.md) for full setup instructions.

**Quick start:**

```bash
cd frontend
npm install
npm start        # Development server at http://localhost:3000
npm run build    # Production build
npm run test     # Run tests (vitest)
npm run lint     # ESLint
npm run prettier # Prettier formatting
```

### Demo mode

Most features depend on microservices deployed in Kubernetes. To develop locally without a live cluster, enable demo mode in `src/config.ts`:

```ts
queryDemo: true
```

Reset to `false` before committing or opening a pull request.

### Source layout

```
src/
├── app/        # App root and routing
├── assets/     # Static assets
├── contexts/   # React contexts
├── features/   # Feature-scoped components (each feature has its own folder)
├── hooks/      # Custom React hooks
├── mocks/      # Mock data for demo mode
├── pages/      # Page-level components
├── services/   # API service clients
├── shared/     # Reusable components used across features
├── static/     # Static files
└── utils/      # Utility functions
```

Each component lives in its own folder with `index.tsx` (implementation) and `index.css` (styles). Tests live alongside their component as `index.test.tsx`.

### Prerequisites

- Node.js ≥ 18 (tested with v23.9.0)
- npm ≥ 10 (tested with v10.9.2)
