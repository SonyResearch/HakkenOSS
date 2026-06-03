# Hypgen Pipeline

This package implements a pipeline to handle the hypothesis generation and selection.

## Installation
To install the package please run:
```bash
uv sync --index-strategy unsafe-best-match
```

Before committing please run:
```bash
uv run mypy .
uv run ruff check
uv run ruff format
```

And check that tests are not failing:
```bash
uv run pytest
```
## Pipeline Implementation

The implementation relies on DVC Data Pipelines. More info can be found here: https://dvc.org/doc/start/data-pipelines/data-pipelines .

First, you need to have dvc installed (globally).

Check that it is linked to the correct remote:
```bash
dvc remote list
# s3      s3://<your-dvc-bucket>
```

The data is stored in `./data`. To get the data run:
```bash
dvc pull
```

Under `data/relations_v2` There is the cleaned file of the knowledge graph that lives on octopus,
namely `edges.csv`. You should add this to the `.env` file (see below) or, always in the `.env`, you 
can provide an alternative path.

### Change pipeline parameters
Each step of the pipeline and its dependencies are defined in the file `dvc.yaml`. There, only
the mandatory parameters are reported. 
All parameters can be changed in the file `params.yaml`.
Let's look at each available step:

- **prepare**:
The parameters to change are the columns names of the raw prediction file, in case
    they differ from the default value:   `node_pair_column`, `node_pair_ocids_column`,
    `predicted_relations_column`, `confidence_scores_column`, `existing_relations_column`
- **filter_by_recency**:
Here the `reference_kg_filepath` must be provided otherwise dvc throws an error, optionally you
    can tune the `median_year` for the filtering (defaults to None), `temporal_popularity_cache` to 
    recompute the median and mode statistics from the reference knowledge graph. If a median year 
    is not provided, no filtering is performed but statistics are added to the output.
- **filter_topk_entities**:
A step to filter out hypothesis involving over-represented entities. Most important parameter to tune is `topk`,
    that defines the number of hypothesis to keep for each node.
- **filter_by_path_length**:
A step to compute the length of the shortest path connecting two nodes in the original knowledge graph. 
    Here we have gain the pointer to the location of the reference knowledge graph `reference_kg_filepath`,
    If the interval for the filtering is provided (`max_path_length` and/or `min_path_length`), the filtering is 
    performed. Otherwise, only the information with the path length is added to the hypothesis file.
- **finalize**:
A step that takes care of polishing the files for delivery. An example is formatting or ordering by confidence score.
The column to use for sorting can be changed by setting `sort_by_column` to a different available column name 
(e.g. 'confidence_score' or 'shortest_path_length'). Also, the nodes are now formatted with the token `<==>`, this can also 
be changed by setting the `format_token` option to a different string value. If not provided, fomatting is via `\s.`

### Run the Pipeline
First add the necessary env variables:
```bash
touch .env
echo "KG_FILEPATH_TSV=your/path/to/the/kg/graph/data" >> .env
echo "KG_STATISTICS_FILEPATH_JSON=your/path/to/the/precomputed/statistics" >> .env # optional, if not there, will be recomputed
source .env
```

To reproduce the pipeline you just need the following command (add `-f` to avoid using the cache):
```bash
dvc repro
```

For the full pipeline, including fetching the hypothesis predictions from s3 and uploading them again on s3,
just run:
```bash
bash run_pipeline.sh
```
NOTE: check the above script first to make sure that the variables are correct(e.g. the batch number you want to generate)
In a future version this code will directly call the API of the core model to generate the hypothesis.

### Run a single step
It is also possible to run independently each single step of the pipeline. The commands are the following:

```bash
uv run prepare <input_hypothesis_file_tsv> <output_folder> <params_file_yaml>
uv run filter-by-recency <input_hypothesis_file_tsv> <output_folder> <params_file_yaml>
uv run filter-topk-entities <input_hypothesis_file_tsv> <output_folder> <params_file_yaml>
uv run filter-by-path-length <input_hypothesis_file_tsv> <output_folder> <params_file_yaml>
uv run finalize <input_hypothesis_file_tsv> <output_folder> <params_file_yaml>
```

# Recover a certain data version
Some specific data versions have been saved in tagged commits, for examples hypothesis sent to DS.
To recover those files do the following:

```bash
git checkout <tag_name>
dvc checkout
```

Available tag names are:
* `ds-hypothesis-v0.0`: The first batch of hypothesis sent to ds 

To go back do:
```bash
git checkout main
dvc checkout
```