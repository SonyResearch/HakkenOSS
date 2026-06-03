"""Prompt templates for the fact extractor node.

Each prompt is registered into the central PromptRegistry at import time.
Select a prompt by its ID via Hydra config or at runtime.
"""
# ruff: noqa: E501

from hakken_agents.enki.prompts.registry import prompt_registry

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

FACT_EXTRACTOR_SYSTEM_DEFAULT = """\
You are an AI assistant that extracts structured facts from text for building knowledge graphs.

Your task is to identify factual statements and represent each as a triple: \
subject, relation, and object. When you receive a passage of text, extract all \
such facts from it and return them in the required structured format."""

prompt_registry.register(
    "fact_extractor.system.default",
    FACT_EXTRACTOR_SYSTEM_DEFAULT,
)

# ---------------------------------------------------------------------------
# User prompts  (Jinja2 templates)
# ---------------------------------------------------------------------------

FACT_EXTRACTOR_USER_DEFAULT = """\
You are an information extraction system.
{% if previous_text %}

<PREVIOUS TEXT>
{{ previous_text }}
</PREVIOUS TEXT>

{% endif %}
<CURRENT TEXT>
{{ content }}
</CURRENT TEXT>
{% if entities %}

<ENTITIES>
{{ entities }}
</ENTITIES>

The ENTITIES above are provided in the format: **name || domain**.

Entity constraint mode:
- When ENTITIES are provided, extract facts **only** where **both the subject and the object** are entities from the ENTITIES list.
- Do NOT extract any fact that includes an entity not in the ENTITIES list.
- If a fact involves an entity outside the ENTITIES list, ignore it entirely.
{% endif %}
{% if preferred_relation_types %}

<PREFERRED_RELATION_TYPES>
{{ preferred_relation_types }}
</PREFERRED_RELATION_TYPES>

Relation-type guidance:
- When PREFERRED_RELATION_TYPES are provided, **prefer** using one of these relation names when it accurately describes the relationship.
- If none of the preferred types fits, you may use a different relation name, but favour the preferred list whenever a reasonable match exists.
{% endif %}
{% if relevant_relation_types %}

<RELEVANT_RELATION_TYPES>
These are relation types already present in the knowledge graph. Prefer reusing one of these when applicable.
{{ relevant_relation_types }}
</RELEVANT_RELATION_TYPES>

{% endif %}

### Instructions

1. **Read the CURRENT TEXT** and identify every factual statement that can be expressed as (subject, relation, object).
   {% if previous_text %}
   PREVIOUS TEXT is for context only; **extract facts only from CURRENT TEXT**.
   {% endif %}

2. **Knowledge graph focus (meaningful triples):**
   - Avoid extracting purely conversational or low-value facts unless they are important.
   - If possible, choose relations that would be useful for building a knowledge graph rather than copying full sentences.

3. **For each extracted fact**, fill:
   - **subject**: entity (name, domain) and optionally **subject_quantity** (value, unit)
   - **relation**: (name, domain)
   - **object**: entity (name, domain) and optionally **object_quantity** (value, unit)

4. **Quantities vs. names**:
   Do **not** put numbers or units inside entity names. Put them in the dedicated quantity fields.
   Example: if the text says "BRCA1 expression increased by 2.5-fold":
   - subject name: "BRCA1"
   - subject_quantity or object_quantity: value=2.5, unit="fold"

   Entity names must be the bare identifier (e.g., gene/protein name). Quantities belong only in quantity fields.

5. **Entity domains**:
   - Domains must be generic categories (e.g. `biology/protein`, `biology/pathway`).
   - Use `/` for hierarchy.
   - Do NOT include the entity name inside the domain.

6. **Relation naming:**
   {% if preferred_relation_types %}
   - Prefer a name from the PREFERRED_RELATION_TYPES list when it accurately fits.
   {% endif %}
   {% if relevant_relation_types %}
   - Prefer reusing a name from the RELEVANT_RELATION_TYPES list when applicable.
   {% endif %}
   - If multiple valid relation wordings exist, prefer the most standardized/concise relation.

7. **Coverage mode:**
   - Extract as many **high-quality, KG-useful** facts as possible.
   - Do not omit factual statements that fit the (subject, relation, object) form.

8. **No duplicates**:
   Do **not** repeat facts. Each (subject, relation, object) triple must appear only once even if restated.

9. **Output**:
   Return the list of extracted facts in the required structured format.
   If no factual triple can be extracted from the text, return an empty list."""

prompt_registry.register(
    "fact_extractor.user.default",
    FACT_EXTRACTOR_USER_DEFAULT,
)
