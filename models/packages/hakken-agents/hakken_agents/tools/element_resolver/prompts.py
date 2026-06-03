"""Jinja2 prompt templates for the element resolver."""

from jinja2 import Template

DESCRIPTION_SYSTEM = """You write short, neutral descriptions of content for semantic search.
Output only the description: one or two sentences that summarize the content.
Do not add preamble or quotes."""

DESCRIPTION_USER = """Content to describe:
{{ content }}"""

_description_user_template: Template = Template(DESCRIPTION_USER)


def render_description_user(content: str) -> str:
    """Render the description user prompt with the given content."""
    return _description_user_template.render(
        content=content.strip(),
    )
