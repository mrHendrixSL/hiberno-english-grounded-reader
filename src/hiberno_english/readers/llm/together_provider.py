"""Together AI transport adapter: turns an already-built `FieldGroupRequest`
into ONE OpenAI-compatible `/chat/completions` call against Together's
serverless endpoint, using Together's schema-constrained Structured Output
facility (`response_format={"type": "json_schema", ...}`) with the EXACT
`RESPONSE_MODELS[group].model_json_schema()` this project already sends to
Ollama's `format` field -- never a looser cloud-specific schema.

Implements the `TransportProvider` protocol from `.execution`, so it plugs
into the SAME `execute_field_group_request`/`execute_field_group_requests`
retry loop and strict-validation call Ollama uses. This module contains no
prompt construction, no retrieval, no resolution, and no B/R/D/G logic --
those are supplied by the caller exactly as for Ollama.

Stdlib `urllib` only, matching this project's existing Ollama transport
(`execution.py`) and Ollama preflight (`scripts/e390_runner/
ollama_preflight.py`) -- no new third-party dependency, no `together` SDK.

The API key is read fresh from the `TOGETHER_API_KEY` environment variable
on every request and is NEVER stored on any dataclass, logged, or included
in any exception message -- see `require_together_api_key`.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping

from .execution import ModelExecutionError, _error_category
from .prompts import FieldGroupRequest

TOGETHER_API_BASE = "https://api.together.xyz/v1"
TOGETHER_API_KEY_ENV_VAR = "TOGETHER_API_KEY"

#: this project's own stdlib-urllib OpenAI-compatible transport, not the
#: official `together` SDK -- recorded truthfully in provenance rather than
#: implying a vendor SDK was used.
TRANSPORT_IMPLEMENTATION = "hiberno-english-together-adapter/1.0.0 (stdlib-urllib)"

#: Together's edge is behind Cloudflare, which blocks urllib's default
#: "Python-urllib/x.y" User-Agent as a bot signature (HTTP 403, Cloudflare
#: error 1010) before the request ever reaches Together's own servers --
#: confirmed live: identical requests succeed with any ordinary-looking
#: User-Agent present and fail without one. Sending a real one here is an
#: infrastructure/transport fix, not a change to any request's scientific
#: content (model, prompts, schema, temperature, seed are untouched).
USER_AGENT = "hiberno-english-together-adapter/1.0.0"


def require_together_api_key() -> str:
    """Returns the current `TOGETHER_API_KEY` value. Raises
    `ModelExecutionError` (never including the key itself, since there is
    none to include when it is missing, and the present value is never
    interpolated into any message) if the environment variable is absent
    or blank. Reads the environment fresh on every call -- the key is
    never cached on any object that could be serialized, logged, or
    dumped into a manifest/provenance record."""
    key = os.environ.get(TOGETHER_API_KEY_ENV_VAR)
    if not key or not key.strip():
        raise ModelExecutionError(
            f"{TOGETHER_API_KEY_ENV_VAR} environment variable is not set -- "
            "refusing to attempt any Together AI request")
    return key


@dataclass(frozen=True)
class TogetherModelExecutionConfig:
    """Experimentally meaningful model/decoding configuration for ONE
    Together AI model -- the cloud analogue of `ModelExecutionConfig`,
    kept as a SEPARATE type rather than reusing it because the two
    providers' non-thinking controls do not share a shape: Ollama turns
    thinking off with a single top-level `think: bool`; each Together
    model exposes its OWN provider-documented control (e.g.
    `chat_template_kwargs={"enable_thinking": False}` for Qwen3.5,
    `reasoning={"enabled": False}` for DeepSeek). `reasoning_control` is
    exactly those extra top-level request keys, supplied by the CALLER
    (never guessed or hard-coded per-model inside this module) and merged
    verbatim into the request payload -- so this module stays generic
    across any Together model.

    `seed` is sent whenever not `None`; whether the endpoint actually
    supports/honours it is a live-verified fact recorded separately in
    provenance (`seed_supported`), never assumed here.
    """

    model_tag: str
    temperature: float = 0.1
    seed: int | None = None
    reasoning_control: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_tag, str) or not self.model_tag.strip():
            raise ModelExecutionError(
                "TogetherModelExecutionConfig.model_tag must be non-empty after strip")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ModelExecutionError("TogetherModelExecutionConfig.temperature must be numeric")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ModelExecutionError(
                "TogetherModelExecutionConfig.temperature must be finite and >= 0")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ModelExecutionError("TogetherModelExecutionConfig.seed must be None or an int")
        if not isinstance(self.reasoning_control, Mapping):
            raise ModelExecutionError(
                "TogetherModelExecutionConfig.reasoning_control must be a mapping")

    def to_dict(self) -> dict:
        return {
            "model_tag": self.model_tag,
            "temperature": self.temperature,
            "seed": self.seed,
            "reasoning_control": dict(self.reasoning_control),
        }

    @staticmethod
    def from_dict(data: dict) -> "TogetherModelExecutionConfig":
        return TogetherModelExecutionConfig(
            model_tag=data["model_tag"],
            temperature=data["temperature"],
            seed=data["seed"],
            reasoning_control=data.get("reasoning_control") or {},
        )


def together_model_execution_sha256(config: TogetherModelExecutionConfig) -> str:
    """Deterministic fingerprint over Together MODEL/DECODING configuration
    only -- the same canonicalization convention as `model_execution_sha256`
    (sorted-key, compact-separator JSON), so the two are directly
    comparable in kind even though they are never mixed into one run."""
    canonical = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TogetherRuntimeConfig:
    """How/where a Together request is transported -- never
    experimentally meaningful, excluded from
    `together_model_execution_sha256` on purpose, exactly mirroring
    `OllamaRuntimeConfig`'s separation of concerns. Never carries the API
    key -- see `require_together_api_key`."""

    base_url: str = TOGETHER_API_BASE
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ModelExecutionError("TogetherRuntimeConfig.base_url must be non-empty")
        if not isinstance(self.timeout_s, (int, float)) or isinstance(self.timeout_s, bool):
            raise ModelExecutionError("TogetherRuntimeConfig.timeout_s must be numeric")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ModelExecutionError("TogetherRuntimeConfig.timeout_s must be finite and > 0")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ModelExecutionError("TogetherRuntimeConfig.max_retries must be an int")
        if self.max_retries < 0:
            raise ModelExecutionError("TogetherRuntimeConfig.max_retries must be >= 0")
        if not isinstance(self.retry_backoff_s, (int, float)) or isinstance(self.retry_backoff_s, bool):
            raise ModelExecutionError("TogetherRuntimeConfig.retry_backoff_s must be numeric")
        if not isfinite(self.retry_backoff_s) or self.retry_backoff_s < 0:
            raise ModelExecutionError(
                "TogetherRuntimeConfig.retry_backoff_s must be finite and >= 0")


def build_together_payload(request: FieldGroupRequest, model_config: TogetherModelExecutionConfig) -> dict:
    """The exact request body sent to `/chat/completions`: the SAME system/
    user prompts `FieldGroupRequest` already carries, and the SAME
    `response_model.model_json_schema()` this project sends to Ollama's
    `format` field -- rendered here as Together's `response_format`
    envelope, never simplified, never with fields removed."""
    payload: dict = {
        "model": model_config.model_tag,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.group,
                "schema": request.response_model.model_json_schema(),
                "strict": True,
            },
        },
        "temperature": model_config.temperature,
        "stream": False,
    }
    if model_config.seed is not None:
        payload["seed"] = model_config.seed
    if model_config.reasoning_control:
        payload.update(model_config.reasoning_control)
    return payload


def _post_chat_completions(base_url: str, payload: dict, timeout_s: float) -> dict:
    """One raw HTTP POST to `{base_url}/chat/completions`. Raises on any
    transport or non-2xx failure; caller applies retry policy. The
    Authorization header carries the API key -- it is never part of
    `payload`, so nothing that dumps `payload` for provenance/telemetry
    can ever include it."""
    api_key = require_together_api_key()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def extract_raw_content(envelope: dict) -> str:
    """`choices[0].message.content` -- markdown-fenced or prose output is
    NEVER stripped/salvaged here; a fenced or non-JSON response fails
    strict validation downstream exactly as a malformed Ollama response
    does."""
    return envelope["choices"][0]["message"]["content"]


def inspect_reasoning_signal(envelope: dict) -> dict:
    """Diagnostic-only (preflight) inspection of whether a response shows
    any sign that reasoning/thinking was NOT actually disabled: a
    provider-native `message.reasoning_content` field, or `content`
    itself starting with a `<think>` tag. Never used by the scientific
    extraction path (which only ever calls `extract_raw_content`) --
    reasoning-disable verification is a preflight concern, not a
    per-request behaviour change."""
    message = {}
    try:
        message = envelope["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        pass
    content = message.get("content") if isinstance(message, dict) else None
    reasoning_content = message.get("reasoning_content") if isinstance(message, dict) else None
    return {
        "reasoning_content_present": bool(reasoning_content),
        "content_starts_with_think_tag": isinstance(content, str) and content.lstrip().startswith("<think>"),
    }


class TogetherTransportProvider:
    """Implements the `TransportProvider` protocol (`.execution`) for
    Together's OpenAI-compatible `/chat/completions` endpoint. Contains
    ONLY transport shape: no prompt text, no retrieval, no resolution, no
    B/R/D/G composition -- those are supplied by the caller exactly as
    for Ollama."""

    def build_payload(self, request: FieldGroupRequest, model_config: TogetherModelExecutionConfig) -> dict:
        return build_together_payload(request, model_config)

    def send(self, payload: dict, runtime_config: TogetherRuntimeConfig) -> tuple[dict, Mapping[str, object] | None]:
        envelope = _post_chat_completions(runtime_config.base_url, payload, runtime_config.timeout_s)
        usage = envelope.get("usage") if isinstance(envelope, dict) else None
        return envelope, usage

    def extract_raw_content(self, envelope: dict) -> str:
        return extract_raw_content(envelope)

    def classify_error(self, exc: BaseException) -> str:
        return _error_category(exc)

    def is_retryable(self, exc: BaseException) -> bool:
        """Only genuinely transient failures are retried: connection
        failures, 429, 5xx, and timeout. A strict-validation failure,
        malformed envelope, or invalid JSON is a DETERMINISTIC mismatch a
        retry cannot fix -- retrying would only spend additional paid
        calls, so these fail immediately (recorded via `classify_error`,
        never repaired, per this task's retry policy)."""
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code == 429 or 500 <= exc.code < 600
        if isinstance(exc, urllib.error.URLError):
            return True
        if isinstance(exc, TimeoutError):
            return True
        return False

    def retry_delay_s(self, exc: BaseException, attempt: int, runtime_config: TogetherRuntimeConfig) -> float:
        """Respects a `Retry-After` header (seconds) when the provider
        supplies one on a 429/5xx; otherwise falls back to the same
        linear `retry_backoff_s * attempt` policy Ollama uses."""
        if isinstance(exc, urllib.error.HTTPError) and exc.headers is not None:
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        return runtime_config.retry_backoff_s * attempt
