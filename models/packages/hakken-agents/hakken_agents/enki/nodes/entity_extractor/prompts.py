"""Prompt templates for the entity extractor node.

Each prompt is registered into the central PromptRegistry at import time.
Select a prompt by its ID via Hydra config or at runtime.
"""
# ruff: noqa: E501

from hakken_agents.enki.prompts.registry import prompt_registry

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ENTITY_EXTRACTOR_SYSTEM_DEFAULT = """\
You are an information extraction system.

Your task is to identify entities and produce reusable, \
context-independent representations suitable for knowledge graphs and indexing."""

prompt_registry.register(
    "entity_extractor.system.default",
    ENTITY_EXTRACTOR_SYSTEM_DEFAULT,
)

# ---------------------------------------------------------------------------
# User prompts  (Jinja2 templates)
# ---------------------------------------------------------------------------

ENTITY_EXTRACTOR_USER_DEFAULT = """\
You are an information extraction system.
{% if allowed_domains %}

<ALLOWED_DOMAINS>
Extracted entities **must** be assigned exactly to one of the following domains. Do not use any other domain.
{{ allowed_domains }}
</ALLOWED_DOMAINS>

{% endif %}
{% if relevant_domains %}

<RELEVANT DOMAINS>
These are existing domains in the knowledge graph. Prefer assigning extracted entities to one of these when applicable.
{{ relevant_domains }}
</RELEVANT DOMAINS>

{% endif %}
{% if previous_text %}

<PREVIOUS TEXT>
{{ previous_text }}
</PREVIOUS TEXT>

{% endif %}
<CURRENT TEXT>
{{ content }}
</CURRENT TEXT>

### Instructions

1. **Entity Identification**
   - Extract all significant entities or concepts that are **explicitly or implicitly** mentioned in the CURRENT TEXT only.
   {% if previous_text %}
   - The PREVIOUS TEXT is provided for context; **do not extract entities** from it.
   {% endif %}

2. **Entity Classification**
   {% if allowed_domains %}
   - Each extracted entity's **domain must be exactly one** of the domains listed in ALLOWED_DOMAINS. Do not assign any other domain.
   {% elif relevant_domains %}
   - Prefer classifying each extracted entity under one of the RELEVANT DOMAINS (existing domains in the knowledge graph) when applicable.
   {% else %}
   - Classify each entity using your best judgment.
   {% endif %}

3. **Domain Assignment**
   - Assign a **domain** to each extracted entity.
   - Domains must be **generic categories**.
   - Domains may be **hierarchical**, using forward slashes (`/`) to indicate increasing specificity.
   - Choose the **most precise applicable domain** without becoming overly specific or idiosyncratic.
   - **Do not include the entity name in the domain.**

4. **Entity Description**
   - For each entity, generate a **context-independent description**.
   - The description must explain what the entity is in general knowledge terms.
   - **Do NOT reference the source text, phrases, mentions, or how the entity appears.**
   - **Do NOT describe relationships to other extracted entities.**
   - The description should remain valid even if the CURRENT TEXT did not exist.


5. **Exclusions**
   - Do NOT extract entities that represent actions or relationships

6. **Formatting Rules**
   - Use **explicit and unambiguous names** for entities (prefer full names when available).
   - Do not infer entities or domains that are not supported by the CURRENT TEXT."""

prompt_registry.register(
    "entity_extractor.user.default",
    ENTITY_EXTRACTOR_USER_DEFAULT,
)

# --- Strict variant: tighter domain rules + temporal exclusions ------------

ENTITY_EXTRACTOR_USER_STRICT = """\
You are an information extraction system.
{% if allowed_domains %}

<ALLOWED_DOMAINS>
Extracted entities **must** be assigned exactly to one of the following domains. Do not use any other domain.
{{ allowed_domains }}
</ALLOWED_DOMAINS>

{% endif %}
{% if relevant_domains %}

<RELEVANT DOMAINS>
These are existing domains in the knowledge graph. Prefer assigning extracted entities to one of these when applicable.
{{ relevant_domains }}
</RELEVANT DOMAINS>

{% endif %}
{% if previous_text %}

<PREVIOUS TEXT>
{{ previous_text }}
</PREVIOUS TEXT>

{% endif %}
<CURRENT TEXT>
{{ content }}
</CURRENT TEXT>

### Instructions

1. **Entity Identification**
   - Extract all significant entities or concepts that are **explicitly or implicitly** mentioned in the CURRENT TEXT only.
   {% if previous_text %}
   - The PREVIOUS TEXT is provided for context; **do not extract entities** from it.
   {% endif %}

2. **Entity Classification**
   {% if allowed_domains %}
   - Each extracted entity's **domain must be exactly one** of the domains listed in ALLOWED_DOMAINS. Do not assign any other domain.
   {% elif relevant_domains %}
   - Prefer classifying each extracted entity under one of the RELEVANT DOMAINS (existing domains in the knowledge graph) when applicable.
   {% else %}
   - Classify each entity using your best judgment.
   {% endif %}

3. **Domain Assignment**
   - Assign a **domain** to each extracted entity.
   - Domains must be **generic categories**, never proper nouns or specific names.
   - Domains may be **hierarchical**, using forward slashes (`/`) to indicate increasing specificity  \
     (e.g., `technology/software`, `biology/genetics`, `finance/investing`).
   - Choose the **most precise applicable domain** without becoming overly specific or idiosyncratic.
   - **Do not include the entity name (or any part of it) in the domain.**
   - If no clear domain can be confidently assigned, leave the domain empty or null.

4. **Entity Description**
   - For each entity, generate a **context-independent description**.
   - The description must explain what the entity is in general knowledge terms.
   - **Do NOT reference the source text, phrases, mentions, or how the entity appears.**
   - **Do NOT describe relationships to other extracted entities.**
   - The description should remain valid even if the CURRENT TEXT did not exist.


5. **Exclusions**
   - Do NOT extract entities that represent:
     - Actions or relationships
     - Dates, times, or other temporal information

6. **Formatting Rules**
   - Use **explicit and unambiguous names** for entities (prefer full names when available).
   - Do not infer entities or domains that are not supported by the CURRENT TEXT."""

prompt_registry.register(
    "entity_extractor.user.strict",
    ENTITY_EXTRACTOR_USER_STRICT,
)
