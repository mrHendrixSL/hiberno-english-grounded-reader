"""Executing already-built FieldGroupRequest objects against a local Ollama
structured-output endpoint, and validating the response strictly against the
five-group contracts.

This module is shared by every LLM reader condition (bare, grounded, and any
future variant): it knows the five field groups' request/response shapes
(via `hiberno_english.readers.llm.contracts`) and nothing else. It has no
knowledge of retrieval, deterministic resolution, precedents, E390, or which
condition produced a given request's prompt content -- that distinction
already lives entirely in the prompt text that 5A's prompt builder produced.

Two configuration objects are kept deliberately separate:

- `ModelExecutionConfig` describes MODEL/DECODING BEHAVIOUR (model tag,
  temperature, seed, think). This is experimentally meaningful: changing it
  changes what is being measured, so it has its own fingerprint
  (`model_execution_sha256`) that depends on nothing else.
- `OllamaRuntimeConfig` describes HOW/WHERE a request is transported (host,
  timeout, retries, backoff). This is a machine/deployment detail: the same
  experiment run against two different hosts must fingerprint identically.

The final research model tag is intentionally NOT hard-coded anywhere in
this module -- it is supplied by the caller and locked elsewhere.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Protocol

from pydantic import BaseModel, ValidationError

from .contracts import FIELD_GROUPS, RESPONSE_MODELS, FieldGroupOutputs
from .prompts import FieldGroupRequest

_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.S)


class ModelExecutionError(RuntimeError):
    """Raised when a field-group request cannot be executed and validated
    against its strict response contract, or when either configuration
    object is malformed.

    Messages here must never reproduce source text, prompts, precedent
    content, or raw model output -- see the module-level docstrings on
    `execute_field_group_request` for what is safe to include."""


# ── configuration: experimental model behaviour ─────────────────────────────
@dataclass(frozen=True)
class ModelExecutionConfig:
    """Experimentally meaningful model/decoding configuration.

    Deliberately excludes anything about how/where the request is
    transported (host, timeout, retries, path) -- see `OllamaRuntimeConfig`
    for that. The same `ModelExecutionConfig` run on two different machines
    must produce the same `model_execution_sha256`.
    """

    model_tag: str
    temperature: float = 0.1
    seed: int | None = None
    think: bool = False

    def __post_init__(self) -> None:
        if not self.model_tag.strip():
            raise ModelExecutionError(
                "ModelExecutionConfig.model_tag must be non-empty after strip")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ModelExecutionError("ModelExecutionConfig.temperature must be numeric")
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ModelExecutionError(
                "ModelExecutionConfig.temperature must be finite and >= 0")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ModelExecutionError("ModelExecutionConfig.seed must be None or an int")
        if not isinstance(self.think, bool):
            raise ModelExecutionError("ModelExecutionConfig.think must be a bool")

    def to_dict(self) -> dict:
        return {
            "model_tag": self.model_tag,
            "temperature": self.temperature,
            "seed": self.seed,
            "think": self.think,
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelExecutionConfig":
        return ModelExecutionConfig(
            model_tag=data["model_tag"],
            temperature=data["temperature"],
            seed=data["seed"],
            think=data["think"],
        )


def model_execution_sha256(config: ModelExecutionConfig) -> str:
    """Deterministic fingerprint over MODEL/DECODING configuration only.

    Never includes host/timeout/retries/backoff -- the same experimental
    configuration run on two machines must fingerprint identically."""
    import hashlib

    canonical = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── model artefact identity ──────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelArtifactIdentity:
    """Immutable identity of the exact local model artefact used for
    execution -- a third, distinct concept from `ModelExecutionConfig`
    (which decoding behaviour was requested) and `OllamaRuntimeConfig`
    (how/where the request was transported). Every field is caller-
    supplied: this type performs no model download, no Ollama API
    request, and no filesystem model-store inspection to obtain them.

    `model_tag` is the exact caller-visible Ollama model name/tag --
    kept here too (not only on `ModelExecutionConfig`) because it is
    part of what identifies THIS artefact; `ExperimentManifest` requires
    it to equal `ModelExecutionConfig.model_tag` at freeze time, while
    the two types themselves stay structurally separate roles.

    `digest` is the model artefact's own immutable SHA-256 identity, as
    reported by the model runtime/store. `quantization` is the exact
    runtime-reported quantisation label -- never inferred from the tag
    or guessed from parameter count.
    """

    model_tag: str
    digest: str
    quantization: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_tag, str) or not self.model_tag.strip():
            raise ModelExecutionError(
                "ModelArtifactIdentity.model_tag must be non-empty after strip")
        if not (isinstance(self.digest, str) and len(self.digest) == 64
               and all(c in "0123456789abcdef" for c in self.digest)):
            raise ModelExecutionError(
                "ModelArtifactIdentity.digest must be a 64-character lowercase "
                f"hex SHA-256 digest, got {self.digest!r}")
        if not isinstance(self.quantization, str) or not self.quantization.strip():
            raise ModelExecutionError(
                "ModelArtifactIdentity.quantization must be non-empty after strip")

    def to_dict(self) -> dict:
        return {
            "model_tag": self.model_tag,
            "digest": self.digest,
            "quantization": self.quantization,
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelArtifactIdentity":
        return ModelArtifactIdentity(
            model_tag=data["model_tag"],
            digest=data["digest"],
            quantization=data["quantization"],
        )


def _rendered_response_contract() -> dict:
    """A freshly-built, JSON-compatible snapshot of the five response
    schemas actually sent to Ollama's `format` field, one per
    `FIELD_GROUPS` entry in canonical order -- never live Pydantic
    internals; a fresh dict on every call."""
    return {group: RESPONSE_MODELS[group].model_json_schema() for group in FIELD_GROUPS}


def response_contract_sha256() -> str:
    """Deterministic fingerprint of the CURRENT five-group structured-
    output contract, exactly as rendered into Ollama's `format` field by
    `_build_payload` below -- not a source-file hash, not a Pydantic
    version string. Ollama receives `response_model.model_json_schema()`
    for each request, so that rendered schema is the actual experimental
    interface a freeze must bind, not the code that produces it.

    Changing any field name, required/optional status, default, Literal
    vocabulary, or strict-extra-forbid behaviour on any of the five
    `RESPONSE_MODELS` changes this digest; changing nothing about them
    reproduces it exactly.
    """
    import hashlib

    canonical = json.dumps(
        _rendered_response_contract(), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── configuration: runtime/transport ────────────────────────────────────────
@dataclass(frozen=True)
class OllamaRuntimeConfig:
    """How/where a request is executed. Never experimentally meaningful --
    excluded from `model_execution_sha256` on purpose."""

    base_url: str = "http://localhost:11434"
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ModelExecutionError("OllamaRuntimeConfig.base_url must be non-empty")
        if not isinstance(self.timeout_s, (int, float)) or isinstance(self.timeout_s, bool):
            raise ModelExecutionError("OllamaRuntimeConfig.timeout_s must be numeric")
        if not isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ModelExecutionError(
                "OllamaRuntimeConfig.timeout_s must be finite and > 0")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ModelExecutionError("OllamaRuntimeConfig.max_retries must be an int")
        if self.max_retries < 0:
            raise ModelExecutionError("OllamaRuntimeConfig.max_retries must be >= 0")
        if not isinstance(self.retry_backoff_s, (int, float)) or isinstance(self.retry_backoff_s, bool):
            raise ModelExecutionError("OllamaRuntimeConfig.retry_backoff_s must be numeric")
        if not isfinite(self.retry_backoff_s) or self.retry_backoff_s < 0:
            raise ModelExecutionError(
                "OllamaRuntimeConfig.retry_backoff_s must be finite and >= 0")


_DEFAULT_RUNTIME = OllamaRuntimeConfig()


# ── think-tag defensive stripping ───────────────────────────────────────────
def _strip_leading_think_block(content: str) -> str:
    """Strip one complete leading `<think>...</think>` block, then trim
    whitespace. Narrow defensive compatibility only: some local models emit
    this despite `think=False` being requested. Does not search for or
    repair anything else -- arbitrary prose is left exactly as received and
    will fail strict validation, not be salvaged."""
    return _THINK_BLOCK.sub("", content, count=1).strip()


# ── the Ollama request contract ─────────────────────────────────────────────
def _build_payload(request: FieldGroupRequest, model_config: ModelExecutionConfig) -> dict:
    options: dict = {"temperature": model_config.temperature}
    if model_config.seed is not None:
        options["seed"] = model_config.seed
    return {
        "model": model_config.model_tag,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        "format": request.response_model.model_json_schema(),
        "options": options,
        "think": model_config.think,
        "stream": False,
    }


def _post_chat(base_url: str, payload: dict, timeout_s: float) -> dict:
    """One raw HTTP POST to `{base_url}/api/chat`. Raises on any transport
    or non-2xx failure; caller applies retry policy."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _extract_and_validate(envelope: dict, response_model: type[BaseModel]) -> BaseModel:
    """Extract `message.content` from an Ollama response envelope and
    validate it strictly against `response_model`. Never regex-extracts
    JSON, never repairs malformed JSON, never coerces prose -- a response
    that does not satisfy the contract as-is fails."""
    content = envelope["message"]["content"]
    content = _strip_leading_think_block(content)
    return response_model.model_validate_json(content)


def _error_category(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_error(status={exc.code})"
    if isinstance(exc, urllib.error.URLError):
        return "transport_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValidationError):
        return "strict_validation_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(exc, KeyError):
        return "malformed_envelope"
    return type(exc).__name__


# ── transport/provider abstraction ──────────────────────────────────────────
# This is the ONLY seam this module exposes for a non-Ollama endpoint (e.g.
# Together AI): a `TransportProvider` knows how to shape ONE already-built
# `FieldGroupRequest` into a raw HTTP call for its endpoint family and how
# to pull raw text back out of the response envelope. It knows nothing
# about prompts, retrieval, resolution, or B/R/D/G -- `execute_field_group_
# request` below still owns the retry loop and the one strict-validation
# call, identically for every provider.
@dataclass(frozen=True)
class CallTelemetry:
    """Optional per-request telemetry, reported exactly once per
    `execute_field_group_request` call (after the final attempt, success
    or exhausted retries) via the `on_call` hook. Every existing Ollama
    caller leaves `on_call` as `None` and is completely unaffected --
    this exists so a paid-API caller (Together) can accumulate token/
    latency/retry telemetry without changing this function's return type
    or its Ollama behaviour."""

    field_group: str
    model_tag: str
    attempts: int
    elapsed_s: float
    error_category: str | None
    usage: Mapping[str, object] | None


class TransportProvider(Protocol):
    """Structural protocol -- a provider need not inherit from this, only
    implement these six methods. `OllamaTransportProvider` below is the
    default, and reproduces this module's pre-existing Ollama behaviour
    exactly."""

    def build_payload(self, request: FieldGroupRequest, model_config: object) -> dict: ...

    def send(self, payload: dict, runtime_config: object) -> tuple[dict, Mapping[str, object] | None]:
        """One raw transport call. Returns `(envelope, usage_or_None)`.
        Raises on transport/HTTP failure -- never retries internally."""

    def extract_raw_content(self, envelope: dict) -> str: ...

    def classify_error(self, exc: BaseException) -> str: ...

    def is_retryable(self, exc: BaseException) -> bool: ...

    def retry_delay_s(self, exc: BaseException, attempt: int, runtime_config: object) -> float: ...


class OllamaTransportProvider:
    """The pre-existing Ollama transport, unchanged, behind the
    `TransportProvider` seam: `build_payload`/`send`/`extract_raw_content`
    delegate to exactly the same `_build_payload`/`_post_chat`/
    `_strip_leading_think_block` this module has always used, and
    `is_retryable`/`retry_delay_s` reproduce the pre-existing
    retry-everything-caught, linear-backoff policy exactly. Wrapping this
    behind the seam changes nothing about what is sent to Ollama or how a
    response is parsed."""

    def build_payload(self, request: FieldGroupRequest, model_config: ModelExecutionConfig) -> dict:
        return _build_payload(request, model_config)

    def send(self, payload: dict, runtime_config: OllamaRuntimeConfig) -> tuple[dict, None]:
        return _post_chat(runtime_config.base_url, payload, runtime_config.timeout_s), None

    def extract_raw_content(self, envelope: dict) -> str:
        return _strip_leading_think_block(envelope["message"]["content"])

    def classify_error(self, exc: BaseException) -> str:
        return _error_category(exc)

    def is_retryable(self, exc: BaseException) -> bool:
        return True

    def retry_delay_s(self, exc: BaseException, attempt: int, runtime_config: OllamaRuntimeConfig) -> float:
        return runtime_config.retry_backoff_s * attempt


_DEFAULT_PROVIDER = OllamaTransportProvider()


def execute_field_group_request(
    request: FieldGroupRequest,
    model_config: object,
    runtime_config: object = _DEFAULT_RUNTIME,
    *,
    provider: TransportProvider | None = None,
    on_call: Callable[[CallTelemetry], None] | None = None,
) -> BaseModel:
    """Execute one `FieldGroupRequest` against `provider`'s endpoint (an
    Ollama `/api/chat` by default) and return an instance of
    `request.response_model`, strictly validated.

    Retries on whatever `provider.classify_error`-catchable failure
    `provider.is_retryable` reports as transient -- up to
    `runtime_config.max_retries + 1` total attempts, sleeping
    `provider.retry_delay_s(...)` between attempts. After exhaustion (or
    immediately, for a non-retryable failure), raises
    `ModelExecutionError`; never returns `None` and never silently
    degrades a failure into an empty lexical response.

    `provider` defaults to `OllamaTransportProvider`, which reproduces
    this function's pre-existing Ollama behaviour exactly -- every
    existing caller that does not pass `provider`/`on_call` is completely
    unaffected by this abstraction.

    The raised error reports only the field group, model tag, attempt
    count, and error category/HTTP status -- never the source entry, the
    prompts, precedent content, or raw model-generated content, any of
    which may carry copyrighted source material.
    """
    if provider is None:
        provider = _DEFAULT_PROVIDER
    model_tag = getattr(model_config, "model_tag", "?")
    payload = provider.build_payload(request, model_config)
    total_attempts = runtime_config.max_retries + 1
    last_category = "unknown"
    start = time.monotonic()

    attempt = 1
    while True:
        try:
            envelope, usage = provider.send(payload, runtime_config)
            content = provider.extract_raw_content(envelope)
            validated = request.response_model.model_validate_json(content)
        except (urllib.error.URLError, TimeoutError, ValidationError,
               KeyError, json.JSONDecodeError, ValueError) as exc:
            last_category = provider.classify_error(exc)
            if attempt < total_attempts and provider.is_retryable(exc):
                time.sleep(provider.retry_delay_s(exc, attempt, runtime_config))
                attempt += 1
                continue
            if on_call is not None:
                on_call(CallTelemetry(
                    field_group=request.group, model_tag=model_tag, attempts=attempt,
                    elapsed_s=time.monotonic() - start, error_category=last_category, usage=None))
            raise ModelExecutionError(
                f"field group '{request.group}' failed after {attempt} "
                f"attempt(s) against model '{model_tag}': {last_category}")
        else:
            if on_call is not None:
                on_call(CallTelemetry(
                    field_group=request.group, model_tag=model_tag, attempts=attempt,
                    elapsed_s=time.monotonic() - start, error_category=None, usage=usage))
            return validated


def execute_field_group_requests(
    requests: Sequence[FieldGroupRequest],
    model_config: object,
    runtime_config: object = _DEFAULT_RUNTIME,
    *,
    provider: TransportProvider | None = None,
    on_call: Callable[[CallTelemetry], None] | None = None,
) -> FieldGroupOutputs:
    """Execute a battery of exactly five `FieldGroupRequest`s, one per
    `FIELD_GROUPS` entry in that exact order, and return one typed
    `FieldGroupOutputs`.

    Validates the request battery itself (count, group identity, order,
    and response-model identity against `RESPONSE_MODELS`) before making
    any HTTP call. If any individual request fails, the underlying
    `ModelExecutionError` propagates immediately; no partial output
    container is ever returned.

    `provider`/`on_call` are forwarded verbatim to every
    `execute_field_group_request` call -- see there for their defaults
    and Ollama-compatibility guarantee.
    """
    if len(requests) != len(FIELD_GROUPS):
        raise ModelExecutionError(
            f"expected {len(FIELD_GROUPS)} field-group requests, got {len(requests)}")

    got_groups = tuple(r.group for r in requests)
    if got_groups != FIELD_GROUPS:
        raise ModelExecutionError(
            f"field-group requests must be in order {FIELD_GROUPS}, got {got_groups}")

    for r in requests:
        expected_model = RESPONSE_MODELS[r.group]
        if r.response_model is not expected_model:
            raise ModelExecutionError(
                f"group '{r.group}' must use response_model "
                f"{expected_model.__name__}, got {r.response_model.__name__}")

    outputs = {
        r.group: execute_field_group_request(
            r, model_config, runtime_config, provider=provider, on_call=on_call)
        for r in requests
    }
    return FieldGroupOutputs(**outputs)
