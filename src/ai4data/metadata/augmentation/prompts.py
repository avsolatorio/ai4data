"""Prompt templates and utilities for data dictionary theme generation.

The prompts follow an elicitation design: the LLM is asked to produce
structured JSON conforming to the ``ThemeGenerationResult`` schema, not
free-form text. This reduces hallucination and makes outputs parseable.

Design choices:
- ``SYSTEM_PROMPT``: Establishes the assistant role and output constraints.
  Explicitly forbids inventing variable names outside the provided list.
- ``USER_PROMPT_TEMPLATE``: Renders the cluster variable list with a numbered
  format so the LLM can reference variables by name easily.
- JSON schema: Derived from ``ThemeGenerationResult.model_json_schema()`` for
  strict type-safe validation at the API boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from .schemas import DictionaryVariable, ThemeGenerationResult

if TYPE_CHECKING:
    pass


# ----- Prompt templates ----- #

SYSTEM_PROMPT = """\
You are a data catalog specialist for social science and development datasets.

GOAL:
Given a list of survey or administrative variable names and labels from a single \
thematic cluster, generate:
1. A concise theme name (2–6 words, title case, no punctuation)
2. A factual 1–2 sentence description grounded in the variable labels provided
3. Up to 5 representative variable names from the INPUT list

CONSTRAINTS:
- Theme name: 2–6 words, title case. Examples: "Household Asset Ownership", \
"Child Health and Nutrition", "Agricultural Land Use".
- Description: factual, 1–2 sentences. Do not speculate beyond the variable labels.
- Example variables: select up to 5 variable names EXACTLY as they appear in the \
INPUT list. Do not invent or paraphrase variable names.
- Output ONLY valid JSON matching the provided schema. No markdown, comments, or \
any text outside the JSON object.
"""

USER_PROMPT_TEMPLATE = """\
# TASK
Generate a theme name and description for the following cluster of survey variables.

# VARIABLES
{variable_list}

Output ONLY valid JSON matching the schema. Do not include markdown or extra text.\
"""


# ----- Token counting ----- #


def count_tokens_approx(text: str) -> int:
    """Approximate token count: words * 1.3, rounded up.

    This is a fast approximation for budget checks. For exact counts, use a
    model-specific tokenizer (e.g., tiktoken for OpenAI models).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    int
        Approximate token count.
    """
    return int(len(text.split()) * 1.3) + 1


# ----- Variable list rendering ----- #


def render_variable_list(variables: List[DictionaryVariable]) -> str:
    """Render a list of variables as a numbered text block for the user prompt.

    Each line has the format:
      ``N. variable_name: Label text — Optional description``

    Parameters
    ----------
    variables : list of DictionaryVariable
        Variables to render.

    Returns
    -------
    str
        Formatted numbered list.
    """
    lines = []
    for i, v in enumerate(variables, 1):
        line = f"{i}. {v.variable_name}: {v.label}"
        if v.description:
            desc = v.description.strip()
            if desc:
                line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


def render_user_prompt(variables: List[DictionaryVariable]) -> str:
    """Render the full user prompt for a cluster.

    Parameters
    ----------
    variables : list of DictionaryVariable
        Cluster variables.

    Returns
    -------
    str
        Formatted user prompt string.
    """
    return USER_PROMPT_TEMPLATE.format(
        variable_list=render_variable_list(variables)
    )


# ----- Response format schema ----- #


def get_theme_response_format() -> Dict:
    """Return the JSON schema dict for structured theme generation responses.

    This is passed as the ``response_format`` argument to litellm / OpenAI
    structured output APIs. The schema is derived from ``ThemeGenerationResult``
    using Pydantic's ``model_json_schema()``, ensuring that the LLM response
    always matches the expected Python type.

    Returns
    -------
    dict
        ``{"type": "json_schema", "json_schema": {...}}`` format.
    """
    schema = ThemeGenerationResult.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "theme_generation",
            "strict": True,
            "schema": schema,
        },
    }


def get_json_object_format() -> Dict:
    """Return a basic JSON object response format for providers without strict schema support.

    Some litellm providers accept ``{"type": "json_object"}`` but not the full
    JSON Schema format. Use this as a fallback when ``get_theme_response_format``
    is not supported.

    Returns
    -------
    dict
    """
    return {"type": "json_object"}
