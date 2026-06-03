# Beam Search Visualizer

[gradio](https://www.gradio.app/) app to show the steps of the beam search.
It is inspired by the following [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) from HuggingFace.

The visualizer code is adapted to work with any search problem (not just token prediction in LLMs) that uses the generic `BeamSearch` implemented in `complex_query/impl/search/beam_search/generic/beam.py`.
In the demo app, the complex querying decomposition problem (`complex_query/impl/search/beam_search/cqd_search.py`) is used for demonstration.

## Steps to Run

- Set config variables in `config.yaml`. Example config is available at `config-example.yaml`.
- Install the package with `playground` dependencies.
  ```shell
  poetry install --with playground
  ```
- Run the application.
  ```shell
  poetry run python app.py
  ```
