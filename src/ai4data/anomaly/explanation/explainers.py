"""Extensible LLM explainer registry for anomaly explanation.

Add new providers by registering a callable that parses a batch output row
into the content dict. Build payloads via llm_client.build_payload (provider-agnostic).
"""

import json
from typing import Any, Callable, Dict

from ai4data.anomaly.explanation.schemas import AnomalyExplanation

# Type for a row parser: (row: dict) -> content_dict or None
RowParser = Callable[[Dict[str, Any]], dict | None]

_EXPLAINER_REGISTRY: Dict[str, RowParser] = {}


def _parse_openai_row(row: Dict[str, Any]) -> dict | None:
    """Parse OpenAI batch output row to content dict."""
    try:
        body = row["response"].get("body", row["response"])
        choices = body.get("choices", [])
        if choices:
            return json.loads(
                choices[0].get("message", {}).get("content", "{}")
            )
        output = body.get("output", [])
        return json.loads(
            output[0].get("content", [{}])[0].get("text", "{}")
        )
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def _parse_gemini_row(row: Dict[str, Any]) -> dict | None:
    """Parse Gemini batch output row to content dict."""
    from pydantic import ValidationError

    try:
        text = (
            row["response"]
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return json.loads(
            AnomalyExplanation.model_validate_json(text).model_dump_json()
        )
    except (ValidationError, json.JSONDecodeError, KeyError):
        return None




def register_explainer(provider: str, parser: RowParser) -> None:
    """Register a new LLM explainer provider.

    Parameters
    ----------
    provider : str
        Provider name (e.g., "openai", "gemini", "anthropic").
    parser : callable
        Function (row: dict) -> content_dict | None.
        Returns the parsed JSON content or None on failure.
    """
    _EXPLAINER_REGISTRY[provider] = parser


def get_explainer(provider: str) -> RowParser | None:
    """Get the row parser for a provider."""
    return _EXPLAINER_REGISTRY.get(provider)


def list_explainers() -> list[str]:
    """List registered explainer provider names."""
    return list(_EXPLAINER_REGISTRY)


# Register built-in providers
register_explainer("openai", _parse_openai_row)
register_explainer("gemini", _parse_gemini_row)
