# llm_client.py
# Centralized Ollama HTTP client.
# All LLM calls go through this module — swapping models later requires changes here only.

import requests
import json
from pydantic import BaseModel
from typing import Type, TypeVar, Optional, Union

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"

# Primary and fallback model names — must match exactly what is pulled in Ollama
PRIMARY_MODEL = "phi4-mini"
FALLBACK_MODEL = "gemma3:2b"

# Default timeout in seconds — LLM calls on CPU can take 2–4 seconds
DEFAULT_TIMEOUT = 120

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Core LLM Call Function
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_message: str,
    response_schema: Optional[Type[T]] = None,
    model: str = PRIMARY_MODEL,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
) -> Union[str, T]:
    """
    Call a local Ollama model and return either raw text or a validated Pydantic object.

    Args:
        system_prompt:    Instructions defining the agent's role and output format.
        user_message:     The input context the agent must reason about.
        response_schema:  If provided, enforces structured JSON output via Ollama's
                          native schema injection. Returns a validated Pydantic object.
        model:            Ollama model name. Defaults to PRIMARY_MODEL (phi4-mini).
        temperature:      Sampling temperature. Lower = more deterministic output.
        timeout:          Request timeout in seconds.

    Returns:
        Validated Pydantic object if response_schema is provided, otherwise raw string.

    Raises:
        requests.exceptions.RequestException: On network or Ollama server errors.
        ValueError: On JSON parse or schema validation failure.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }

    # Inject JSON schema into payload for structured output enforcement
    if response_schema:
        payload["format"] = response_schema.model_json_schema()

    response = requests.post(OLLAMA_BASE_URL, json=payload, timeout=timeout)
    response.raise_for_status()

    content = response.json()["message"]["content"]

    if response_schema:
        return response_schema.model_validate_json(content)

    return content


# ---------------------------------------------------------------------------
# Model Health Check
# ---------------------------------------------------------------------------

def ping_model(model: str) -> bool:
    """
    Verify a model is loaded and responding in Ollama.
    Returns True if the model responds, False otherwise.
    Call this during startup before running the pipeline.
    """
    try:
        response = requests.post(
            OLLAMA_BASE_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return "READY" in content.upper()
    except Exception:
        return False


def verify_models() -> dict:
    """
    Check both primary and fallback models are available.
    Returns a status dict for each model.
    Call this at startup in main.py before pipeline execution.
    """
    return {
        PRIMARY_MODEL: ping_model(PRIMARY_MODEL),
        FALLBACK_MODEL: ping_model(FALLBACK_MODEL),
    }
