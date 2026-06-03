"""Jinja2 prompt templates for the domain resolver."""

from jinja2 import Template

DESCRIPTION_SYSTEM = """You write short, neutral descriptions of domain names or concepts
for semantic search. The description should capture what the domain represents
(e.g. topic, category, or type).
Output only the description: one or two sentences.
Do not add preamble or quotes."""

DESCRIPTION_USER = """{% if context %}Context: {{ context }}

{% endif %}Domain name or concept to describe:
{{ content }}"""

_description_user_template: Template = Template(DESCRIPTION_USER)


def render_description_user(content: str, context: str | None = None) -> str:
    """Render the description user prompt with the given domain name and optional context."""
    return _description_user_template.render(
        content=content.strip(),
        context=(context.strip() if context else None),
    )
