"""LLM client utilities for building API payloads."""

from typing import Any, Dict


ENDPOINT_URLS = {
    "completions": "/v1/chat/completions",
    "responses": "/v1/responses",
}


def build_payload(
    endpoint: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    response_format: dict,
    with_search: bool = False,
    **api_kwargs: Any,
) -> Dict[str, Any]:
    """Build an API payload for anomaly explanation using an LLM.

    Parameters
    ----------
    endpoint : str
        One of "completions" or "responses".
    model_id : str
        Model identifier (e.g., "gpt-4.1-mini").
    system_prompt : str
        System message content.
    user_prompt : str
        User message content (with context rendered).
    response_format : dict
        JSON schema for structured output (e.g., from prompts.get_anomaly_response_format).
    with_search : bool
        Whether to enable web search tool (OpenAI only, not supported in batch).
    **api_kwargs
        Overrides for default API parameters.

    Returns
    -------
    dict
        Payload suitable for the specified endpoint.
    """
    if endpoint not in ENDPOINT_URLS:
        raise ValueError(f"endpoint must be one of {list(ENDPOINT_URLS)}")

    default_kwargs: Dict[str, Any] = {
        "model": model_id,
        "temperature": 0,
        "max_completion_tokens": 8192,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "seed": 1029,
        "store": True,
    }
    default_kwargs.update(api_kwargs)

    if with_search:
        if "tools" not in default_kwargs:
            default_kwargs["tools"] = [
                {"type": "web_search", "search_context_size": "low"}
            ]
        else:
            tools = default_kwargs["tools"]
            if not any(t.get("type") == "web_search" for t in tools):
                default_kwargs["tools"] = tools + [
                    {"type": "web_search", "search_context_size": "low"}
                ]
        if "include" not in default_kwargs:
            default_kwargs["include"] = ["web_search_call.action.sources"]
        elif "web_search_call.action.sources" not in default_kwargs["include"]:
            default_kwargs["include"].append("web_search_call.action.sources")
        default_kwargs["tool_choice"] = "required"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    if endpoint == "responses":
        for msg in messages:
            if msg["content"]:
                msg["content"][0]["type"] = "input_text"
        payload = {
            "input": messages,
            "text": {"format": {"type": "json_schema", **response_format["json_schema"]}},
            **{k: v for k, v in default_kwargs.items() if k not in ("frequency_penalty", "presence_penalty", "seed")},
        }
        if "max_completion_tokens" in payload:
            payload["max_output_tokens"] = payload.pop("max_completion_tokens")
    else:
        payload = {
            "messages": messages,
            "response_format": response_format,
            **default_kwargs,
        }

    return payload
