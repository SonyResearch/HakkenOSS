# API Endpoints

The Hakken Models API provides RESTful endpoints for interacting with THiGER models. All endpoints are prefixed with `/thiger`.

## Base URL

By default, the service runs on `http://localhost:8088`. All endpoints are under the `/thiger` prefix.

## Authentication

Currently, the API does not require authentication. In production, configure authentication through the `spaice-inference-api` framework.

## Endpoints

### Predict Relations

Predict relations between entity pairs.

**Endpoint:** `POST /thiger/predict`

**Request Body:**

```json
{
  "subject_id_list": ["entity_id_1", "entity_id_2"],
  "object_id_list": ["entity_id_3", "entity_id_4"],
  "relation_id_list": ["relation_1", "relation_2"],  // Optional
  "inference_config": {  // Optional
    "on_missing": "skip",  // or "raise"
    "normalize": true
  }
}
```

**Response:**

```json
{
  "relations_ids": ["relation_1", "relation_2"],
  "relations_probs": [
    [0.1, 0.9],  // Probabilities for first entity pair
    [0.3, 0.7]   // Probabilities for second entity pair
  ],
  "relations_scores": [
    [-2.3, 2.1],  // Raw scores for first entity pair
    [-1.2, 1.5]   // Raw scores for second entity pair
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8088/thiger/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id_list": ["Chemical|MESH:C494910"],
    "object_id_list": ["Gene|3553"],
    "relation_id_list": ["negative_correlate"]
  }'
```

### Score Facts

Score existing facts (triples) using the model.

**Endpoint:** `POST /thiger/score`

**Request Body:**

```json
{
  "facts_list": [
    ["subject_id", "relation_id", "object_id"],
    ["subject_id_2", "relation_id_2", "object_id_2"]
  ],
  "inference_config": {  // Optional
    "on_missing": "skip",
    "normalize": true
  }
}
```

**Response:**

```json
{
  "scores_list": [2.3, -1.5],
  "normalized_scores_list": [0.9, 0.1]
}
```

**Example:**

```bash
curl -X POST http://localhost:8088/thiger/score \
  -H "Content-Type: application/json" \
  -d '{
    "facts_list": [
      ["Chemical|MESH:C494910", "negative_correlate", "Gene|3553"]
    ]
  }'
```

### Get Entity Pair Indexes

Map entity IDs to internal indexes and get entity pair tensor representation.

**Endpoint:** `POST /thiger/entity-pair-indexes`

**Request Body:**

```json
{
  "subject_id_list": ["entity_id_1", "entity_id_2"],
  "object_id_list": ["entity_id_3", "entity_id_4"],
  "inference_config": {  // Optional
    "on_missing": "skip"
  }
}
```

**Response:**

```json
{
  "subject_index_list": [0, 1],
  "object_index_list": [2, 3],
  "entity_pairs": [[0, 2], [1, 3]]
}
```

### Get Fact Indexes

Map fact IDs to internal indexes.

**Endpoint:** `POST /thiger/fact-indexes`

**Request Body:**

```json
{
  "facts_list": [
    ["subject_id", "relation_id", "object_id"]
  ],
  "inference_config": {  // Optional
    "on_missing": "skip"
  }
}
```

**Response:**

```json
{
  "fact_index_list": [
    [0, 1, 2]  // [subject_idx, relation_idx, object_idx]
  ]
}
```

### Sample Random Facts

Sample random facts from dataset splits.

**Endpoint:** `POST /thiger/sample-facts`

**Request Body:**

```json
{
  "num_samples": 10,
  "splits": ["train", "val", "test"]  // Default: ["train"]
}
```

**Response:**

```json
{
  "facts_list": [
    ["subject_id_1", "relation_id_1", "object_id_1"],
    ["subject_id_2", "relation_id_2", "object_id_2"]
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8088/thiger/sample-facts \
  -H "Content-Type: application/json" \
  -d '{
    "num_samples": 5,
    "splits": ["train", "test"]
  }'
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request format
- `500 Internal Server Error` - Server error (includes error details in response)

Error response format:

```json
{
  "detail": "Error message or traceback"
}
```

## Inference Configuration

The `inference_config` parameter (optional) allows you to customize inference behavior:

- `on_missing`: How to handle missing entities/relations
  - `"skip"` - Skip missing items (default)
  - `"raise"` - Raise an error
- `normalize`: Whether to normalize scores to probabilities (default: `true`)
