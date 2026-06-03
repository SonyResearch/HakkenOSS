# Extract embeddings

- RDF2KG conversion
- Use of pyrdf2vec methodology

### Steps to reproduce

1. `uv sync --index-strategy unsafe-best-match`
2. Check `uv run mypy && flake8 && pytest`
2. Run `uv run embeddings --ontology_file owl_file_example.nt --entities_file entities_file_example.txt --output_file ./output/embs.json`

### Warning 
- The current pyrdf2vec version is very slow with the weisfeiler lehman walker, so the default was set to the random walker. 