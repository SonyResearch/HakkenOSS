# Hakken Models API

Welcome to the Hakken Models API documentation. This service provides a RESTful interface for serving predictive models (e.g., THiGER), enabling knowledge graph completion and inference operations.

## What is Hakken Models API?

Hakken Models API is a FastAPI-based service that wraps THiGER models to provide:

- **Prediction**: Predict relations between entity pairs
- **Scoring**: Score facts (subject-relation-object triples)
- **Index Mapping**: Convert between entity/fact IDs and internal indexes
- **Fact Sampling**: Sample random facts from dataset splits

The service is built on the `spaice-inference-api` framework, which provides a standardized interface for serving machine learning models with dependency injection, logging, and model lifecycle management.

## Key Features

- 🚀 **FastAPI-based**: Modern, async REST API
- 🔧 **Flexible Model Loading**: Support for MLflow and directory-based model loading
- 📊 **THiGER Integration**: Seamless integration with Hakken Models package
- 🔌 **Dependency Injection**: Clean architecture with dependency-injector
- 📝 **Type Safety**: Full Pydantic models for request/response validation
- 🐳 **Docker Support**: Containerized deployment ready

## Quick Links

- [Getting Started](getting-started.md) - Installation and basic setup
- [API Reference](api/endpoints.md) - Complete endpoint documentation
- [Configuration](configuration.md) - Configuration options and examples
- [Architecture](architecture.md) - System architecture and design

## Requirements

- Python 3.11-3.12
- PyTorch 2.4+ (with CUDA support recommended)
- `hakken_models` package
- `spaice-inference-api` framework

## Getting Help

For issues, questions, or contributions, please refer to the project repository or contact the maintainers.

