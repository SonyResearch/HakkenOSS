"""Enki prompt registry — single entry-point for all prompt look-ups.

Usage::

    from hakken_agents.enki.prompts import prompt_registry

    system = prompt_registry.get("entity_extractor.system.default")

Runtime registration::

    prompt_registry.register("my_custom.system.v1", "You are ...")
"""

# Trigger built-in prompt registration by importing the per-node modules.
import hakken_agents.enki.nodes.entity_extractor.prompts as _ee_prompts  # noqa: F401
import hakken_agents.enki.nodes.fact_extractor.prompts as _fe_prompts  # noqa: F401

from .registry import PromptRegistry, prompt_registry

__all__ = ["PromptRegistry", "prompt_registry"]
