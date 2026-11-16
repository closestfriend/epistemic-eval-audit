"""Shared LLM judge client and utilities for native target scorers."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_JUDGE_MODEL = os.environ.get("AUDIT_JUDGE_MODEL", "gemma-4-31b-it")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_judge_sync(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    use_cache: bool = True,
) -> str:
    """Call the configured LLM judge synchronously.

    Uses Adaption (`ADAPTION_API_KEY`) as primary backend with disk caching.
    Falls back to OpenRouter or OpenAI only if ADAPTION_API_KEY is unset.
    """
    model_name = model or DEFAULT_JUDGE_MODEL

    if os.environ.get("ADAPTION_API_KEY"):
        from harness.adaption_provider import get_adaption_client

        res = get_adaption_client().generate_sync(
            model=model_name,
            prompt=prompt,
            instructions=system_message,
            temperature=temperature,
            max_output_tokens=max_tokens,
            use_cache=use_cache,
        )
        return res["text"]

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        url = OPENROUTER_URL
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }
    elif openai_key:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        }
    else:
        raise RuntimeError(
            "No API key found. Set ADAPTION_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        )

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Exception | None = None
    for _ in range(3):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=45.0)
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Judge call failed after 3 attempts: {last_err}")
