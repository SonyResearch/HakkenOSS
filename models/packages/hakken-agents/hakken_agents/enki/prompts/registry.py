from __future__ import annotations


class PromptRegistry:
    """Central registry for prompt templates, keyed by string ID.

    Supports both build-time registration (module-level calls in prompt files)
    and runtime registration (user-defined prompts).
    """

    def __init__(self) -> None:
        self._prompts: dict[str, str] = {}

    def register(self, prompt_id: str, template: str) -> None:
        """Register a prompt template. Raises if the ID is already taken."""
        if prompt_id in self._prompts:
            raise ValueError(f"Prompt '{prompt_id}' is already registered")
        self._prompts[prompt_id] = template

    def override(self, prompt_id: str, template: str) -> None:
        """Register or replace a prompt template (for runtime patching)."""
        self._prompts[prompt_id] = template

    def get(self, prompt_id: str) -> str:
        """Retrieve a prompt template by ID."""
        try:
            return self._prompts[prompt_id]
        except KeyError:
            available = ", ".join(sorted(self._prompts))
            raise KeyError(f"Prompt '{prompt_id}' not found. Available: {available}") from None

    def has(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    def list_prompts(self) -> list[str]:
        return sorted(self._prompts)


prompt_registry = PromptRegistry()
