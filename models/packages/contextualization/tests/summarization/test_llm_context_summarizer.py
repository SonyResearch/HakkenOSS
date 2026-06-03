import pytest
from transformers import AutoConfig, AutoModel, AutoTokenizer

from contextualization.core.entities.config.context_summarizer import (
    LLMContextSummarizerConfig,
)
from contextualization.core.entities.publication import Publication
from contextualization.core.entities.retrieval import Reference
from contextualization.impl.context_summarizer import LLMContextSummarizer


@pytest.fixture
def references(ndjson_publications_path) -> list[Reference]:
    publications = []
    with open(ndjson_publications_path) as f:
        for line in f:
            publications.append(Publication.model_validate_json(line))

    references = []
    for i in range(10):
        references.append(
            Reference(
                publication_info=publications[i % len(publications)],
                score=0.5,
                text=f"reference text {i}",
            )
        )

    return references


@pytest.fixture
def test_model_path(tmp_path):
    cache_dir = tmp_path / "cache"

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=cache_dir)
    config.hidden_size = 24
    config.intermediate_size = 32
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.layer_types = config.layer_types[:2]
    model = AutoModel.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=cache_dir)

    model.save_pretrained(str(tmp_path))
    tokenizer.save_pretrained(str(tmp_path))

    return str(tmp_path)


class TestLLMContextSummarizer:
    def test_summarize_all_references(self, test_model_path, references):
        summarizer_config = LLMContextSummarizerConfig(
            hf_model_name_or_path=test_model_path,
            prompt_all_references="all references prompt",
            prompt_single_reference="single reference prompt",
            device="cpu",
            max_new_tokens=16,
        )
        summarizer = LLMContextSummarizer(config=summarizer_config)
        summarizer.summarize_all_references(references)

    def test_summarize_reference(self, test_model_path, references):
        summarizer_config = LLMContextSummarizerConfig(
            hf_model_name_or_path=test_model_path,
            prompt_all_references="all references prompt",
            prompt_single_reference="single reference prompt",
            device="cpu",
            max_new_tokens=16,
        )
        summarizer = LLMContextSummarizer(config=summarizer_config)
        summarizer.summarize_reference(references[0])
