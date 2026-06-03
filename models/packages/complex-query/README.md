# Complex Query

A package for processing complex DNF queries over knowledge graphs.

## Container Configuration

- See `QueryingSettings` in `complex_query/container.py` for required values.
  - It will read either from environment variables or `.env` file.
  - Variables will be read case insensitively.
  - Nested values can be provided by using `__` (two underscores) as delimiter.
    - For example, the `host` value of `neo4j_config` can be set by the environment variable `NEO4J_CONFIG__HOST`.
- See `tests/test_container.py` and `tests/test_container.env` for example code with the `QueryingContainer`.

## Running the Querying Server

- See `/services/query-interface` (in the project root) for details of running the querying interface service.
- Please also see `complex_query/delivery/rest/router.py` for information about available requests.

## Schemas

### Input

#### Note

- Concepts and relations in a query should be specified by OCIDs of string type.
- Legacy format (will be deprecated from `0.4.0` & removed from `1.0.0`)
  - OCIDs for concepts and relations are prepended with the prefix `id`, e.g. `id150000002942` for the concept with OCID `150000002942`.
  - Domains are represented in OCID, as specified in `complex_query/core/values/domain.py`, e.g. `id210000003484`, `210000003484` (integer), or `'210000003484'` (string) for the domain `ANATOMY`.
    e.g. `id210000003484` for the domain `ANATOMY`.
  - Search algorithm is specified in a query (e.g. `"beam"`); this will be ignored.
- Preferred format from `0.4.0`
  - OCIDs are given without the `id` prefix, e.g. `'150000002942'` (string).
  - Domains are given directly by their names represented in [SCREAMING_SNAKE_CASE](https://en.wikipedia.org/wiki/Snake_case) or their original names, e.g. `'ANATOMY'` or `'anatomy'`, `'EFFECTS_PROCESSES_AND_FUNCTIONS'` or `'effects, processes, and functions'` (see also `complex_query/core/values/domain.py` for the recognizable domain names).
  - Search algorithm is not specified in a query; instead is specified in the container setting.
- The formula should be in [Disjunctive Normal Form (DNF)](https://en.wikipedia.org/wiki/Disjunctive_normal_form).

#### Legacy format (or before `0.4.0`)

```json
{
  "variables": [
    {
      "label": "x",
      "domain": "id229940000407"
    }
  ],
  "formula": "P(x, id232000000216, id150000002942) AND P(x, id232000000087, id241001148079)",
  "n_candidates": 2,
  "search_algorithm": "beam",
  "search_parameters": {
    "beam_size": 5
  }
}
```

#### Preferred format (from `0.4.0`)

```json
{
  "variables": [
    {
      "label": "x",
      "domain": "DRUG_RELATED_CONCEPT"
    }
  ],
  "formula": "P(x, '232000000216', '150000002942') AND P(x, '232000000087', '241001148079')",
  "n_candidates": 2,
  "search_parameters": {
    "beam_size": 5
  }
}
```

### Output

```json
{
  "candidates": [
    {
      "var_assignments": {
        "x": "229940063215"
      },
      "condition_scores": {
        1: 0.9999247940500176,
        2: 0.17581747657523306
      },
      "query_score": 0.17580425405
    },
    {
      "var_assignments": {
        "x": "229940082632"
      },
      "condition_scores": {
        1: 0.999987018700965,
        2: 0.12813251187363395
      },
      "query_score": 0.12813084854
    }
  ]
}
```
