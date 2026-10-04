"""Minimal OpenRouter chat client (stdlib only) with retries and JSON parsing."""
from __future__ import annotations

import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# Any OpenAI-compatible endpoint works. Set LLM_BASE_URL=http://localhost:11434/v1 for a local Ollama server.
# Several comma-separated URLs are used in rotation, so two vLLM servers on two GPUs share the load.
BASE_URLS = [u.strip() for u in os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").split(",") if u.strip()]
BASE_URL = BASE_URLS[0]
_counter = itertools.count()


def load_key() -> str:
    if not BASE_URL.startswith("https://openrouter.ai"):
        return os.environ.get("LLM_API_KEY", "none")
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for d in [Path.cwd(), *Path.cwd().parents]:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("OPENROUTER_API_KEY not set and no .env found")


def chat(model: str, system: str, user: str, *, temperature: float = 0.7,
         max_tokens: int = 1500, retries: int = 4, base_urls: list[str] | None = None) -> tuple[str, dict]:
    """Return (text, usage). Raises after exhausting retries. `base_urls` overrides LLM_BASE_URL for this call (to send
    a request to the server that hosts `model`)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **({"reasoning_effort": os.environ["LLM_REASONING"]} if os.environ.get("LLM_REASONING") else {}),
        # Extra request fields as JSON, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for vLLM.
        **json.loads(os.environ.get("LLM_EXTRA", "{}")),
    }).encode()
    headers = {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            urls = base_urls or BASE_URLS
            url = urls[next(_counter) % len(urls)].rstrip("/") + "/chat/completions"
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("generation hit max_tokens")
            return choice["message"]["content"], data.get("usage", {})
        except (urllib.error.URLError, KeyError, RuntimeError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed after {retries} attempts: {last}")


def parse_json(text: str):
    """Parse a JSON object or array, tolerating code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))
