# Request/Response Models

This page documents the Pydantic models used for API requests and responses.

## Request Models

### ModelPredictRequest

Request model for predicting relations between entity pairs.

```python
class ModelPredictRequest(BaseModel):
    subject_id_list: list[str]
    object_id_list: list[str]
    relation_id_list: list[str] | None = None
    inference_config: dict | None = None
```

**Fields:**
- `subject_id_list` (required): List of subject entity IDs
- `object_id_list` (required): List of object entity IDs (must match length of `subject_id_list`)
- `relation_id_list` (optional): List of relation IDs to filter predictions. If `None`, predicts for all relations
- `inference_config` (optional): Dictionary of inference parameters (see [Inference Configuration](#inference-configuration))

### ModelScoreRequest

Request model for scoring facts.

```python
class ModelScoreRequest(BaseModel):
    facts_list: list[tuple[str, str, str]]
    inference_config: dict | None = None
```

**Fields:**
- `facts_list` (required): List of facts represented as tuples of `(subject_id, relation_id, object_id)`
- `inference_config` (optional): Dictionary of inference parameters

### EntityPairIndexRequest

Request model for mapping entity pairs to indexes.

```python
class EntityPairIndexRequest(BaseModel):
    subject_id_list: list[str]
    object_id_list: list[str]
    inference_config: dict | None = None
```

**Fields:**
- `subject_id_list` (required): List of subject entity IDs
- `object_id_list` (required): List of object entity IDs
- `inference_config` (optional): Dictionary of inference parameters

### FactIndexRequest

Request model for mapping facts to indexes.

```python
class FactIndexRequest(BaseModel):
    facts_list: list[FactType]  # FactType = tuple[str, str, str]
    inference_config: dict | None = None
```

**Fields:**
- `facts_list` (required): List of facts as `(subject_id, relation_id, object_id)` tuples
- `inference_config` (optional): Dictionary of inference parameters

### SampleFactsRequest

Request model for sampling random facts.

```python
class SampleFactsRequest(BaseModel):
    num_samples: int = 10
    splits: list[Literal["train", "val", "test"]] = ["train"]
```

**Fields:**
- `num_samples` (optional, default: 10): Number of facts to sample
- `splits` (optional, default: `["train"]`): Dataset splits to sample from

## Response Models

### ModelPredictResponse

Response model for relation predictions.

```python
class ModelPredictResponse(BaseModel):
    relations_ids: list[str]
    relations_probs: list[list[float]] | None
    relations_scores: list[list[float]] | None
```

**Fields:**
- `relations_ids`: The relation IDs for which scores were computed
- `relations_probs`: Normalized probability scores. `relations_probs[i][j]` is the probability for relation `j` and entity pair `i`. `None` if normalization is disabled
- `relations_scores`: Raw model scores. `relations_scores[i][j]` is the raw score for relation `j` and entity pair `i`

**Structure:**
- Outer list length = number of entity pairs
- Inner list length = number of relations

### ModelScoreResponse

Response model for fact scoring.

```python
class ModelScoreResponse(BaseModel):
    scores_list: list[float]
    normalized_scores_list: list[float] | None
```

**Fields:**
- `scores_list`: Raw model scores for each input fact
- `normalized_scores_list`: Normalized scores (probabilities) for each input fact. `None` if normalization is disabled

### EntityPairIndexResponse

Response model for entity pair index mapping.

```python
class EntityPairIndexResponse(BaseModel):
    subject_index_list: list[int]
    object_index_list: list[int]
    entity_pairs: list[tuple[int, int]]
```

**Fields:**
- `subject_index_list`: Internal indexes for subject entities
- `object_index_list`: Internal indexes for object entities
- `entity_pairs`: Entity pair tensor representation as list of `(subject_idx, object_idx)` tuples

### FactIndexResponse

Response model for fact index mapping.

```python
class FactIndexResponse(BaseModel):
    fact_index_list: list[FactIndexType]  # FactIndexType = tuple[int, int, int]
```

**Fields:**
- `fact_index_list`: List of facts as `(subject_idx, relation_idx, object_idx)` tuples

### SampleFactsResponse

Response model for sampled facts.

```python
class SampleFactsResponse(BaseModel):
    facts_list: list[FactType]  # FactType = tuple[str, str, str]
```

**Fields:**
- `facts_list`: List of sampled facts as `(subject_id, relation_id, object_id)` tuples

## Inference Configuration

The `inference_config` dictionary supports the following parameters:

```python
{
    "on_missing": "skip" | "raise",  # How to handle missing entities/relations
    "normalize": bool                 # Whether to normalize scores to probabilities
}
```

**Parameters:**
- `on_missing`: 
  - `"skip"` - Skip missing entities/relations and continue processing
  - `"raise"` - Raise an error if any entity/relation is missing
- `normalize`: 
  - `true` - Return normalized probabilities in addition to raw scores
  - `false` - Return only raw scores

## Type Definitions

```python
FactType = tuple[str, str, str]  # (subject_id, relation_id, object_id)
FactIndexType = tuple[int, int, int]  # (subject_idx, relation_idx, object_idx)
```

## Validation

All models use Pydantic for validation:

- Type checking is enforced
- Required fields must be provided
- Optional fields have default values
- Invalid data will return a `400 Bad Request` error with validation details

## Example Usage

```python
from hakken_models_api.entities.predict import ModelPredictRequest, ModelPredictResponse

# Create request
request = ModelPredictRequest(
    subject_id_list=["Chemical|MESH:C494910"],
    object_id_list=["Gene|3553"],
    relation_id_list=["negative_correlate"],
    inference_config={"on_missing": "skip", "normalize": True}
)

# Request is automatically validated
# Invalid data would raise ValidationError
```

