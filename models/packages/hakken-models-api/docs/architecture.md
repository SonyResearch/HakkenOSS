# Architecture

This document describes the architecture and design of the Hakken Models API service.

## Overview

The Hakken Models API is built on top of the `spaice-inference-api` framework, which provides a standardized interface for serving machine learning models. The service uses FastAPI for the REST API, dependency injection for clean architecture, and Hydra for configuration management.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
│                  (spaice-inference-api)                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Router Layer                         │
│           (hakken_models_api/routers/)                  │
│  - /thiger/predict                                      │
│  - /thiger/score                                        │
│  - /thiger/entity-pair-indexes                         │
│  - /thiger/fact-indexes                                 │
│  - /thiger/sample-facts                                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Dependency Injection Container             │
│            (hakken_models_api/container.py)             │
│  - Configuration                                        │
│  - Model Artifacts                                      │
│  - Logger                                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Model Loader Layer                      │
│            (hakken_models_api/loaders/)                 │
│  - THiGERRunLoader                                      │
│  - Loads from MLflow or Directory                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Hakken Models Package                  │
│              (hakken_models.models.thiger)              │
│  - THiGERArtifacts                                      │
│  - THiGERLoader                                         │
│  - Dataset                                              │
│  - Model (PyTorch)                                      │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### Service Entry Point

**File:** `service.py`

The service entry point uses Hydra to load configuration and initializes the dependency injection container. It creates a FastAPI server using `spaice-inference-api`'s `create_server` function.

**Key responsibilities:**
- Load configuration via Hydra
- Initialize dependency injection container
- Wire dependencies
- Create and start FastAPI server

### Router Layer

**Directory:** `hakken_models_api/routers/`

The routers define API endpoints using FastAPI's `APIRouter`. Each endpoint:
- Uses dependency injection to get required services
- Validates requests using Pydantic models
- Calls model functions from `hakken_models`
- Returns validated responses

**Endpoints:**
- `POST /thiger/predict` - Predict relations for entity pairs
- `POST /thiger/score` - Score facts
- `POST /thiger/entity-pair-indexes` - Map entity IDs to indexes
- `POST /thiger/fact-indexes` - Map fact IDs to indexes
- `POST /thiger/sample-facts` - Sample random facts

### Dependency Injection

**File:** `hakken_models_api/container.py`

Uses `dependency-injector` to manage dependencies. The container provides:
- Configuration (`HakkenModelsAPIConfig`)
- Model artifacts (via `ModelToken` from `spaice-inference-api`)
- Logger (via `LoggerToken` from `spaice-inference-api`)

### Model Loader

**Directory:** `hakken_models_api/loaders/`

Implements `IModelLoader` interface from `spaice-inference-api`. The loaders:
- Supports loading from MLflow or directory
- Uses `THiGERLoader` from `hakken_models` package
- Returns `THiGERArtifacts` containing model and dataset

**Loading modes:**
1. **MLflow**: Loads from MLflow tracking server using run ID
2. **Directory**: Loads from local or S3 directory

### Configuration

**File:** `hakken_models_api/config.py`

Defines `HakkenModelsAPIConfig` using Pydantic Settings:
- Validates that exactly one loading mode is specified
- Provides type-safe configuration access
- Supports environment variable overrides

### Entity Models

**Directory:** `hakken_models_api/entities/`

Pydantic models for request/response validation:
- `predict.py` - Predict request/response models
- `score.py` - Score request/response models
- `data.py` - Index mapping and sampling models


## Design Patterns

### Dependency Injection

All dependencies are injected via `dependency-injector`:
- Eliminates global state
- Enables easy testing (mock dependencies)
- Provides clear dependency graph

### Configuration Management

Uses Hydra for configuration:
- YAML-based configuration
- Environment variable overrides
- Validation via Pydantic

### Request/Response Validation

All API models use Pydantic:
- Automatic validation
- Type safety
- Clear error messages

### Model Lifecycle

Managed by `spaice-inference-api`:
- Model loaded once at startup
- Cached in memory
- Shared across requests
- Thread-safe access

## Error Handling

Errors are handled at multiple levels:

1. **Validation Errors**: Pydantic validates requests, returns `400 Bad Request`
2. **Business Logic Errors**: Caught in router, returns `500 Internal Server Error` with details
3. **Model Errors**: Propagated from `hakken_models`, wrapped in HTTP exceptions

## Threading and Concurrency

- FastAPI handles requests asynchronously
- Model inference uses `@torch.no_grad()` for efficiency
- Model artifacts are shared and thread-safe
- PyTorch models handle concurrent inference automatically

## Extensibility

The architecture supports extension:

1. **New Endpoints**: Add routes to the appropriate router in `routers/`
2. **New Models**: Implement `IModelLoader` interface
3. **New Configurations**: Extend `HakkenModelsAPIConfig`
4. **Custom Logic**: Add utility functions in `utils.py`

## Dependencies

### Core Dependencies
- `fastapi` - Web framework
- `spaice-inference-api` - Inference framework
- `hakken_models` - Model implementations
- `pydantic` - Data validation
- `dependency-injector` - Dependency injection
- `hydra-core` - Configuration management

### Model Dependencies
- `torch` - PyTorch
- `pytorch-lightning` - Lightning framework (for checkpoints)

## Performance Considerations

- Models are loaded once and cached
- GPU inference for faster predictions
- Batch processing supported (multiple entity pairs)
- `@torch.no_grad()` decorator for inference efficiency

## Security Considerations

- Input validation via Pydantic
- No authentication currently (add via `spaice-inference-api` in production)
- Error messages don't expose internal paths
- S3 credentials managed via AWS SDK

