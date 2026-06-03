from typing import TYPE_CHECKING

from transformers import pipeline

from contextualization.core.contracts.context_summarizer import ContextSummarizer
from contextualization.core.entities.config.context_summarizer import (
    LLMContextSummarizerConfig,
)
from contextualization.core.entities.retrieval import Reference
from contextualization.core.entities.summarization import PromptMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from contextualization.core.entities.retrieval import Reference


class LLMContextSummarizer(ContextSummarizer[LLMContextSummarizerConfig]):
    def __init__(self, config: LLMContextSummarizerConfig) -> None:
        super().__init__(config)

        self.summary_generator = pipeline(
            task="text-generation", model=config.hf_model_name_or_path
        )

    def summarize_all_references(self, references: "Sequence[Reference]") -> str:
        references_text = "\n\n".join(
            f"<id>{i}</id>"
            f"<title>{reference.publication_info.title}</title>\n"
            f"<abstract>{reference.publication_info.abstract}</abstract>"
            for i, reference in enumerate(references, 1)
        )
        messages = [PromptMessage(role="system", content=self.config.prompt_all_references)]
        messages.append(PromptMessage(role="user", content=references_text))
        outputs = self.summary_generator(
            messages,  # type: ignore
            max_new_tokens=self.config.max_new_tokens,
        )
        return outputs[0]["generated_text"][-1]["content"]  # type: ignore

    def summarize_reference(self, reference: "Reference") -> str:
        reference_text = (
            f"<title>{reference.publication_info.title}</title>\n"
            f"<abstract>{reference.publication_info.abstract}</abstract>"
        )
        messages = [PromptMessage(role="system", content=self.config.prompt_single_reference)]
        messages.append(PromptMessage(role="user", content=reference_text))
        outputs = self.summary_generator(
            messages,  # type: ignore
            max_new_tokens=self.config.max_new_tokens,
        )
        return outputs[0]["generated_text"][-1]["content"]  # type: ignore
