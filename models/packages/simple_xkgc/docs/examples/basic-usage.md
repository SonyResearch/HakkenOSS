# Basic Usage Examples

## Example 1: Simple Explanation

Generate explanations for a single triple using default settings:

```python
from hakken_explainer.explainers import HakkenExplainer
from hakken_explainer.candidate_finder.corpus import CorpusPathFinder
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.constants import ScoreType

# Setup
candidate_finder = CorpusPathFinder(max_candidates=1000)
candidate_finder.setup(facts_batch=search_space, kg=kg, kge=kge)

explainer = HakkenExplainer(
    candidate_finder=candidate_finder,
    model=gnn_model,
    kg=kg,
    search_space=search_space
)

# Generate explanations
score_configs = [ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=32)]
explanations = explainer.explain(
    triple_to_probe=("entity_1", "relation_1", "entity_2"),
    device="cuda",
    explanation_length=2,
    score_type_list=score_configs
)

print(explanations.head())
```

## Example 2: Using Both Scoring Methods

Compute both sufficient and necessary scores:

```python
from hakken_explainer.entities.config import ScoreTypeConfig
from hakken_explainer.constants import ScoreType

score_configs = [
    ScoreTypeConfig(type=ScoreType.SUFFICIENT, batch_size=32),
    ScoreTypeConfig(type=ScoreType.NECESSARY, batch_size=8)
]

explanations = explainer.explain(
    triple_to_probe=triple,
    score_type_list=score_configs
)

# Access individual scores
print(explanations[['explanation', 'score_sufficient', 'score_necessary', 'score']])
```

## Example 3: Filtering by Relations

Only find paths that use specific relations:

```python
allowed_relations_ids = ["relation_1", "relation_2"]

explanations = explainer.explain(
    triple_to_probe=triple,
    allowed_relations_ids=allowed_relations_ids
)
```

## Example 4: Using Latent Space Finder

Use KGE embeddings to find paths:

```python
from hakken_explainer.candidate_finder.latent.kge import KGEPathCandidateFinder
from hakken_explainer.candidate_finder.path_generator.kge import KGEPathGenerator

path_generator = KGEPathGenerator(kge_model_path=kge_folder)
candidate_finder = KGEPathCandidateFinder(
    max_candidates=5000,
    path_generator=path_generator
)
candidate_finder.setup(facts_batch=search_space, kg=kg, kge=kge)
```

## Example 5: Custom Path Length

Specify a custom explanation length:

```python
# Find paths of length 3 (3 edges)
explanations = explainer.explain(
    triple_to_probe=triple,
    explanation_length=3
)
```

## Example 6: Saving Results

Save explanations to a file:

```python
from hakken_ml_toolkit.ml_utils import DSVUtils
from pathlib import Path

DSVUtils.write_dsv(
    df=explanations,
    file_path=Path("explanations.tsv"),
    delimiter="\t"
)
```

