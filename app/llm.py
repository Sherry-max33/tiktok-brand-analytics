"""Minimal OpenAI chat client (standard library only) for the AI layer.

The key stays server-side: it is read from the environment, Streamlit Secrets, or the
repo's gitignored .env, and is only ever sent to the OpenAI API. OPENAI_MODEL overrides the
default model. Returns parsed JSON from a strict JSON-schema response, or raises LLMError.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import catalog

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1"
TIMEOUT_S = 45
DEFAULT_RETRIES = 0


class LLMError(RuntimeError):
    pass


def _env_file_value(name: str) -> str:
    path = catalog.REPO_ROOT / ".env"
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def setting(name: str) -> str:
    """Environment, then Streamlit Secrets, then the local .env."""
    if os.getenv(name):
        return os.environ[name]
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:  # no secrets file, or not running under Streamlit
        pass
    return _env_file_value(name)


def api_key() -> str:
    return setting("OPENAI_API_KEY")


def model_name() -> str:
    return setting("OPENAI_MODEL") or DEFAULT_MODEL


USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}  # per process, for cost reports


def available() -> bool:
    return bool(api_key())


def _post(body: dict, key: str, retries: int) -> dict:
    """POST; retries (with backoff) only on timeouts, rate limits and server errors."""
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        wait = 2 ** (attempt + 1)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise LLMError(f"OpenAI HTTP {error.code}: {error.read()[:300]!r}") from error
            try:
                wait = max(wait, float(error.headers.get("retry-after") or 0))
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries:
                raise LLMError(f"OpenAI request failed: {error}") from error
        time.sleep(min(wait, 30))
    raise LLMError("unreachable")


def complete_json(system: str, user: str, schema: dict, *, schema_name: str) -> dict:
    """One chat completion. Live (public) generation never retries a failed call
    (DEFAULT_RETRIES = 0); offline curation scripts may raise DEFAULT_RETRIES."""
    retries = DEFAULT_RETRIES
    key = api_key()
    if not key:
        raise LLMError("OPENAI_API_KEY is not set")
    body = {
        "model": model_name(),
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    payload = _post(body, key, retries)
    usage = payload.get("usage") or {}
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
    USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
    try:
        message = payload["choices"][0]["message"]
        if message.get("refusal"):
            raise LLMError(f"Model refused: {message['refusal']}")
        return json.loads(message["content"])
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        raise LLMError("Unexpected OpenAI response shape") from error
