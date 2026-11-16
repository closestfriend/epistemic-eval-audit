"""Adaption Inference Provider with disk caching and Inspect AI ModelAPI registration.

Wraps the official Adaption Python SDK (`adaption>=0.16.0`) via
`Adaption().responses.create(...)` with retry-safe execution, deterministic
temperature/seed controls, and JSONL disk caching under `results/cache/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from adaption import Adaption, AsyncAdaption
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "cache"
DEFAULT_MODEL = os.environ.get("AUDIT_JUDGE_MODEL", "gemma-4-31b-it")
REASONING_MODELS = frozenset({
    "qwen3.5-397b",
    "glm-5.2",
    "glm-53-flash",
    "glm-5.3-flash",
    "deepseek-v4-flash",
    "qwen3.8-27b",
})
MIN_REASONING_TOKENS = 2048
_LEGACY_TOKEN_BUDGETS = (128, 256, 300, 1024, 2048)

_CACHE_LOCK = threading.Lock()
_MEM_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}
_THREAD_FLAGS = threading.local()


def _effective_max_tokens(model: str, max_output_tokens: int) -> int:
    """Enforce a >= 2048 output budget on reasoning models to prevent empty truncations."""
    if model in REASONING_MODELS:
        return max(max_output_tokens, MIN_REASONING_TOKENS)
    return max_output_tokens


def _extract_message_text(resp: Any) -> str:
    """Extract reply text strictly from `message` items in `resp.output`, never `reasoning`."""
    parts: list[str] = []
    for item in getattr(resp, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content_part in getattr(item, "content", None) or []:
            part_type = getattr(content_part, "type", None)
            if part_type == "output_text":
                txt = getattr(content_part, "text", None)
                if txt:
                    parts.append(txt)
            elif part_type == "refusal":
                ref = getattr(content_part, "refusal", None)
                if ref:
                    parts.append(ref)
    return "".join(parts).strip()


class bypass_cache:
    """Context manager to temporarily disable AdaptionClient disk cache on current thread."""

    def __enter__(self) -> None:
        _THREAD_FLAGS.bypass = True

    def __exit__(self, *args: Any) -> None:
        _THREAD_FLAGS.bypass = False


def _ensure_cache_loaded(cache_file: Path) -> dict[str, dict[str, Any]]:
    if cache_file not in _MEM_CACHE:
        index: dict[str, dict[str, Any]] = {}
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        k = rec.get("key")
                        if k and str(rec.get("text", "")).strip():
                            index[k] = rec
                    except json.JSONDecodeError:
                        continue
        _MEM_CACHE[cache_file] = index
    return _MEM_CACHE[cache_file]


def _cache_key(
    model: str,
    instructions: str,
    input_payload: Any,
    temperature: float,
    max_output_tokens: int,
) -> str:
    raw = json.dumps(
        {
            "model": model,
            "instructions": instructions,
            "input": input_payload,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache_entry(cache_file: Path, key: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        index = _ensure_cache_loaded(cache_file)
        entry = index.get(key)
        if entry is not None and str(entry.get("text", "")).strip():
            return entry
        return None


def _lookup_cached(
    cache_file: Path,
    model: str,
    instructions: str,
    prompt: Any,
    temperature: float,
    max_output_tokens: int,
    effective_max_tokens: int,
) -> dict[str, Any] | None:
    candidate_budgets = dict.fromkeys([
        effective_max_tokens,
        max_output_tokens,
        *_LEGACY_TOKEN_BUDGETS,
    ])
    with _CACHE_LOCK:
        index = _ensure_cache_loaded(cache_file)
        for budget in candidate_budgets:
            key = _cache_key(model, instructions, prompt, temperature, budget)
            entry = index.get(key)
            if entry is not None and str(entry.get("text", "")).strip():
                return entry
        return None


def _save_cache_entry(cache_file: Path, entry: dict[str, Any]) -> None:
    if not str(entry.get("text", "")).strip():
        return
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        index = _ensure_cache_loaded(cache_file)
        index[entry["key"]] = entry
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_valid_entry(key: str, model: str, resp: Any) -> dict[str, Any]:
    """Validate response completion and non-empty message text, returning a cache entry."""
    text = _extract_message_text(resp)
    status = getattr(resp, "status", None)
    if status != "completed" or not text:
        raise RuntimeError(
            f"Incomplete/empty response from {model}: status={status!r}, len(text)={len(text)}"
        )
    usage = getattr(resp, "usage", None)
    return {
        "key": key,
        "model": model,
        "status": status,
        "text": text,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "cost": getattr(usage, "cost", 0.0) if usage else 0.0,
    }


class AdaptionClient:
    """Retry-safe, disk-cached wrapper around Adaption().responses.create."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("ADAPTION_API_KEY")
        self._thread_local = threading.local()
        self._async_client: AsyncAdaption | None = None
        self._cache_dir = Path(cache_dir)
        self._use_cache = use_cache

    def _get_sync_client(self) -> Adaption:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = (
                Adaption(api_key=self._api_key, timeout=60.0)
                if self._api_key
                else Adaption(timeout=60.0)
            )
            self._thread_local.client = client
        return client

    def _cache_file_for(self, model: str) -> Path:
        safe_model = model.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe_model}.jsonl"

    def _cache_enabled(self, use_cache: bool | None) -> bool:
        return (self._use_cache if use_cache is None else use_cache) and not getattr(
            _THREAD_FLAGS, "bypass", False
        )

    def generate_sync(
        self,
        model: str,
        prompt: str | list[dict[str, str]],
        instructions: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """Synchronously call Adaption responses.create with caching and retries."""
        cache_enabled = self._cache_enabled(use_cache)
        effective_tokens = _effective_max_tokens(model, max_output_tokens)
        key = _cache_key(model, instructions, prompt, temperature, effective_tokens)
        cache_file = self._cache_file_for(model)

        if cache_enabled:
            cached = _lookup_cached(
                cache_file,
                model,
                instructions,
                prompt,
                temperature,
                max_output_tokens,
                effective_tokens,
            )
            if cached is not None:
                return cached

        req_max_tokens = effective_tokens
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = self._get_sync_client().responses.create(
                    model=model,
                    instructions=instructions,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=req_max_tokens,
                    store=False,
                )
                entry = _extract_valid_entry(key, model, resp)
                if cache_enabled:
                    _save_cache_entry(cache_file, entry)
                return entry
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if "Incomplete/empty response" in str(exc):
                    req_max_tokens = max(req_max_tokens * 2, 4096)
                time.sleep(min(2.0**attempt, 16.0))
        raise RuntimeError(f"Adaption call failed after 5 attempts ({model}): {last_err}")

    async def generate_async(
        self,
        model: str,
        prompt: str | list[dict[str, str]],
        instructions: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """Asynchronously call Adaption responses.create with caching and retries."""
        import asyncio

        cache_enabled = self._cache_enabled(use_cache)
        effective_tokens = _effective_max_tokens(model, max_output_tokens)
        key = _cache_key(model, instructions, prompt, temperature, effective_tokens)
        cache_file = self._cache_file_for(model)

        if cache_enabled:
            cached = _lookup_cached(
                cache_file,
                model,
                instructions,
                prompt,
                temperature,
                max_output_tokens,
                effective_tokens,
            )
            if cached is not None:
                return cached

        if self._async_client is None:
            self._async_client = (
                AsyncAdaption(api_key=self._api_key, timeout=60.0)
                if self._api_key
                else AsyncAdaption(timeout=60.0)
            )

        req_max_tokens = effective_tokens
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = await self._async_client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=req_max_tokens,
                    store=False,
                )
                entry = _extract_valid_entry(key, model, resp)
                if cache_enabled:
                    _save_cache_entry(cache_file, entry)
                return entry
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if "Incomplete/empty response" in str(exc):
                    req_max_tokens = max(req_max_tokens * 2, 4096)
                await asyncio.sleep(min(2.0**attempt, 16.0))
        raise RuntimeError(f"Async Adaption call failed after 5 attempts ({model}): {last_err}")


_SHARED_CLIENT: AdaptionClient | None = None


def get_adaption_client() -> AdaptionClient:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None:
        _SHARED_CLIENT = AdaptionClient()
    return _SHARED_CLIENT


@modelapi(name="adaption")
class AdaptionModelAPI(ModelAPI):
    """Inspect AI ModelAPI provider for Adaption (`adaption/<model_id>`)."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["ADAPTION_API_KEY"],
            config=config,
        )
        self._client = AdaptionClient(api_key=self.api_key)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        instructions_parts: list[str] = []
        convo: list[dict[str, str]] = []
        for msg in input:
            text = msg.text if hasattr(msg, "text") else str(msg.content)
            if msg.role == "system":
                instructions_parts.append(text)
            elif msg.role in ("user", "assistant"):
                convo.append({"role": msg.role, "content": text})

        instructions = "\n\n".join(instructions_parts) or "You are a helpful assistant."
        prompt_input: Any = (
            convo[0]["content"]
            if len(convo) == 1 and convo[0]["role"] == "user"
            else convo
        )
        temperature = config.temperature if config.temperature is not None else 0.0
        max_tokens = config.max_tokens if config.max_tokens is not None else 1024

        res = await self._client.generate_async(
            model=self.model_name,
            prompt=prompt_input,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        out = ModelOutput.from_content(
            model=f"adaption/{self.model_name}",
            content=res["text"],
        )
        out.usage = ModelUsage(
            input_tokens=res.get("input_tokens", 0),
            output_tokens=res.get("output_tokens", 0),
            total_tokens=res.get("input_tokens", 0) + res.get("output_tokens", 0),
        )
        return out
