"""Composing the four held-out LLM reader conditions -- B, R, D, G -- from
components that already exist elsewhere in this package.

    B = bare generation (no precedents)
    R = retrieved generation (precedents supplied)
    D = rho(B)   -- deterministic resolution applied to B's own output
    G = rho(R)   -- deterministic resolution applied to R's own output

There are exactly TWO raw language-model generation batteries per source
entry -- bare and retrieved -- never four. D and G are never independently
generated: they are `hiberno_english.resolution.resolve_entry` applied to
B's and R's own already-assembled `Entry` objects. This module is the
first layer in this package allowed to know that condition design; the
prompt/execution/assembly layer it calls (`prompts.py`, `execution.py`,
`assembly.py`) still knows nothing about B/R/D/G, and the resolution layer
(`hiberno_english.resolution`) still knows nothing about retrieval or
model execution.

This module performs no retrieval of its own -- it consumes an
already-bound `QueryPrecedents` (from `hiberno_english.readers.llm.
precedents`) and never imports the retrieval-mathematics or bank-binding
modules that produced it. It performs no corpus-wide orchestration,
scoring, or integrity diagnostics; those belong to a later build.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ...resolution import DeterministicResolutionContext, resolve_entry
from ...schema import Entry, Provenance
from .assembly import assemble_entry
from .execution import (
    CallTelemetry,
    ModelExecutionConfig,
    OllamaRuntimeConfig,
    TransportProvider,
    execute_field_group_requests,
    model_execution_sha256,
)
from .precedents import QueryPrecedents
from .prompts import build_field_group_requests

_DEFAULT_RUNTIME = OllamaRuntimeConfig()


class FactorialReaderError(RuntimeError):
    """Raised when an `LLMEntryInput`, a supplied `QueryPrecedents`, or the
    B/R inputs to factorial composition fail an experimental precondition,
    or when composed D/G output violates the annotation-only resolution
    invariant. Raised before any HTTP request where the violation is
    detectable beforehand; no partial `FactorialReaderOutputs` is ever
    returned alongside this exception.
    """


@dataclass(frozen=True)
class LLMEntryInput:
    """One already-segmented lexical entry prepared by a later workflow,
    with no retrieval data, model configuration, resolution context, or
    output path attached -- those are supplied separately to the functions
    that consume this object."""

    entry_id: str
    entry_type: str
    homonym_index: int | None
    source_entry: str
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id:
            raise FactorialReaderError(
                f"LLMEntryInput.entry_id must be a non-empty string, got {self.entry_id!r}")
        if not isinstance(self.entry_type, str) or not self.entry_type:
            raise FactorialReaderError(
                f"LLMEntryInput.entry_type must be a non-empty string, got {self.entry_type!r}")
        if not isinstance(self.source_entry, str) or not self.source_entry:
            raise FactorialReaderError(
                "LLMEntryInput.source_entry must be a non-empty string, "
                f"got {self.source_entry!r}")
        if self.homonym_index is not None and (
               isinstance(self.homonym_index, bool) or not isinstance(self.homonym_index, int)):
            raise FactorialReaderError(
                "LLMEntryInput.homonym_index must be None or a plain int, "
                f"got {self.homonym_index!r}")
        if not isinstance(self.provenance, Provenance):
            raise FactorialReaderError(
                "LLMEntryInput.provenance must be a hiberno_english.schema.Provenance, "
                f"got {type(self.provenance).__name__}")


def _decided_by(config_sha256: str, *, retrieval: bool) -> str:
    """Deterministic generation-provenance label: an already-computed
    model-config fingerprint plus the retrieval factor -- never a
    filesystem-oriented run name. Never includes the resolution factor:
    D's and G's generated leaves are not independently generated, so they
    retain B's and R's own `decided_by` values respectively.

    Takes the fingerprint itself, not a `ModelExecutionConfig`, so this
    label works identically for any provider's own config/fingerprint
    type (e.g. Together's `TogetherModelExecutionConfig` and
    `together_model_execution_sha256`) -- see `run_factorial_entry`'s
    `config_fingerprint` parameter."""
    return f"llm:{config_sha256}:retrieval={'on' if retrieval else 'off'}"


def generate_llm_entry(
    entry_input: LLMEntryInput,
    *,
    authority_context: Mapping[str, object],
    precedents: Sequence[Entry],
    model_config: object,
    runtime_config: object = _DEFAULT_RUNTIME,
    decided_by: str,
    provider: TransportProvider | None = None,
    on_call: Callable[[CallTelemetry], None] | None = None,
) -> Entry:
    """One raw generation battery: five field-group requests, executed,
    assembled into one `Entry`. No resolution, no retrieval, no condition
    branching -- `precedents` is used exactly as supplied, and this
    function never calls `resolve_entry`.

    `entry_input.provenance` is deep-copied before assembly, so the
    returned `Entry` never shares a mutable `Provenance` object with
    `entry_input` or with any other `Entry` this function produces.

    `model_config`/`runtime_config` are duck-typed, not required to be
    `ModelExecutionConfig`/`OllamaRuntimeConfig` -- any object
    `execute_field_group_requests`'s configured `provider` understands is
    accepted (e.g. Together's `TogetherModelExecutionConfig`/
    `TogetherRuntimeConfig`), so this function itself never needs to know
    which provider is in play. `provider`/`on_call` default to Ollama's
    existing behaviour -- see `execution.execute_field_group_request`.
    """
    requests = build_field_group_requests(
        source_entry=entry_input.source_entry,
        authority_context=authority_context,
        precedents=precedents,
    )
    outputs = execute_field_group_requests(
        requests, model_config, runtime_config, provider=provider, on_call=on_call)
    return assemble_entry(
        entry_id=entry_input.entry_id,
        entry_type=entry_input.entry_type,
        homonym_index=entry_input.homonym_index,
        outputs=outputs,
        provenance=entry_input.provenance.model_copy(deep=True),
        decided_by=decided_by,
    )


@dataclass(frozen=True)
class FactorialReaderOutputs:
    """The four held-out LLM reader conditions for one source entry.
    Uppercase field names correspond exactly to the condition identifiers
    used in the experiment and manuscript."""

    B: Entry
    R: Entry
    D: Entry
    G: Entry


def _masked_dump(entry: Entry) -> dict:
    """`entry.model_dump` with every resolution-only field reset to its
    unresolved default, for comparing two Entry objects while ignoring
    exactly the fields rho is permitted to change."""
    d = entry.model_dump(mode="json", exclude_none=False)
    for s in d.get("senses", []):
        for r in s.get("regions", []):
            r["resolved_name"] = None
            r["source_type"] = None
        for g in s.get("glosses", []):
            for r in g.get("regions", []):
                r["resolved_name"] = None
                r["source_type"] = None
    for vf in d.get("variant_forms", []):
        for r in vf.get("regions", []):
            r["resolved_name"] = None
            r["source_type"] = None
    for x in d.get("cross_references", []):
        x["resolution"] = None
        x["targets"] = []
        x["resolves"] = False
    return d


def _require_same_source_identity(bare_entry: Entry, retrieved_entry: Entry) -> None:
    if bare_entry.entry_id != retrieved_entry.entry_id:
        raise FactorialReaderError(
            "bare_entry and retrieved_entry entry_id disagree: "
            f"{bare_entry.entry_id!r} != {retrieved_entry.entry_id!r}")
    if bare_entry.entry_type != retrieved_entry.entry_type:
        raise FactorialReaderError(
            "bare_entry and retrieved_entry entry_type disagree: "
            f"{bare_entry.entry_type!r} != {retrieved_entry.entry_type!r}")
    if bare_entry.homonym_index != retrieved_entry.homonym_index:
        raise FactorialReaderError(
            "bare_entry and retrieved_entry homonym_index disagree: "
            f"{bare_entry.homonym_index!r} != {retrieved_entry.homonym_index!r}")
    if bare_entry.provenance.witness_id != retrieved_entry.provenance.witness_id:
        raise FactorialReaderError(
            "bare_entry and retrieved_entry provenance.witness_id disagree "
            "-- these are not two readings of the same source document "
            f"revision: {bare_entry.provenance.witness_id!r} != "
            f"{retrieved_entry.provenance.witness_id!r}")


def _require_only_resolution_changed(before: Entry, after: Entry, *, label: str) -> None:
    if _masked_dump(before) != _masked_dump(after):
        raise FactorialReaderError(
            f"resolution changed a non-resolution field going into {label} "
            "-- this violates the annotation-only B->D / R->G invariant")


def compose_factorial_outputs(
    *,
    bare_entry: Entry,
    retrieved_entry: Entry,
    resolution_context: DeterministicResolutionContext,
) -> FactorialReaderOutputs:
    """Pure composition: no LLM call, no prompt construction, no
    retrieval. `B`/`R` are deep copies of the supplied entries; `D`/`G`
    are `resolve_entry(B, ...)`/`resolve_entry(R, ...)` -- never an
    independent generation.

    Requires `bare_entry` and `retrieved_entry` to be two readings of the
    same source entry (`entry_id`, `entry_type`, `homonym_index`, and
    `provenance.witness_id` must all agree) but never requires their
    lexical content to agree -- retrieval may legitimately change what a
    reader generates. After composing D and G, re-verifies (even though
    `resolve_entry` already guarantees it) that only the permitted
    resolution fields differ from B/R respectively, since this is the
    experimental boundary the whole B/R/D/G contrast depends on. Raises
    `FactorialReaderError` on any violation.
    """
    _require_same_source_identity(bare_entry, retrieved_entry)

    B = bare_entry.model_copy(deep=True)
    R = retrieved_entry.model_copy(deep=True)

    D = resolve_entry(B, resolution_context)
    G = resolve_entry(R, resolution_context)

    _require_only_resolution_changed(B, D, label="D")
    _require_only_resolution_changed(R, G, label="G")

    return FactorialReaderOutputs(B=B, R=R, D=D, G=G)


def _validate_query_precedents(entry_input: LLMEntryInput, retrieved_precedents: QueryPrecedents) -> None:
    """Recheck the experimental preconditions on an already-bound
    `QueryPrecedents` before spending any model call. `QueryPrecedents`
    itself already enforces internal consistency (matching lengths,
    matching entry_ids); this additionally ties it to THIS entry_input and
    rejects a self-precedent and an empty precedent set, which are
    experimental requirements this module owns, not `precedents.py`'s."""
    if retrieved_precedents.query_entry_id != entry_input.entry_id:
        raise FactorialReaderError(
            "retrieved_precedents.query_entry_id "
            f"{retrieved_precedents.query_entry_id!r} does not match "
            f"entry_input.entry_id {entry_input.entry_id!r}")
    if not retrieved_precedents.precedents:
        raise FactorialReaderError(
            "retrieved_precedents must be non-empty for the R branch")

    pids = retrieved_precedents.precedent_entry_ids
    if len(set(pids)) != len(pids):
        raise FactorialReaderError(f"duplicate precedent_entry_id(s): {pids}")

    for pid, precedent in zip(pids, retrieved_precedents.precedents):
        if precedent.entry_id != pid:
            raise FactorialReaderError(
                f"precedent Entry.entry_id {precedent.entry_id!r} does not "
                f"match its precedent_entry_ids value {pid!r}")

    if entry_input.entry_id in pids:
        raise FactorialReaderError(
            "retrieved_precedents contains the query's own entry_id "
            f"{entry_input.entry_id!r}")


def run_factorial_entry(
    entry_input: LLMEntryInput,
    *,
    authority_context: Mapping[str, object],
    retrieved_precedents: QueryPrecedents,
    model_config: object,
    resolution_context: DeterministicResolutionContext,
    runtime_config: object = _DEFAULT_RUNTIME,
    provider: TransportProvider | None = None,
    on_call: Callable[[CallTelemetry], None] | None = None,
    config_fingerprint: Callable[[object], str] = model_execution_sha256,
) -> FactorialReaderOutputs:
    """Run exactly two generation batteries (bare, retrieved) for one
    source entry and compose all four B/R/D/G conditions from them --
    exactly ten model requests, never twenty.

    One `model_config`, one `runtime_config`, one `authority_context`, and
    one `entry_input` are used for both the bare and the retrieved
    battery; the ONLY generation-level difference between them is whether
    `precedents` is empty or `retrieved_precedents.precedents` -- the
    controlled retrieval contrast this experiment measures.

    `provider`/`on_call` default to Ollama's existing behaviour (see
    `execution.execute_field_group_request`); `config_fingerprint`
    defaults to `model_execution_sha256`, so every existing caller that
    passes only a `ModelExecutionConfig` and does not pass these three
    keyword arguments gets IDENTICAL behaviour to before this parameter
    set existed. A different provider supplies its own config type and
    its own fingerprint function (e.g. Together's
    `TogetherModelExecutionConfig`/`together_model_execution_sha256`)
    without this module needing to know anything about it beyond that it
    is callable.

    Validates `retrieved_precedents` against `entry_input` before any HTTP
    request; on failure raises `FactorialReaderError` and performs zero
    requests. If either generation battery fails, the underlying
    `ModelExecutionError` propagates immediately -- no partial or
    fabricated `FactorialReaderOutputs` is ever produced, and this
    function never substitutes an empty or heuristic value for a failed
    field group.
    """
    _validate_query_precedents(entry_input, retrieved_precedents)

    digest = config_fingerprint(model_config)
    off_label = _decided_by(digest, retrieval=False)
    on_label = _decided_by(digest, retrieval=True)

    bare_entry = generate_llm_entry(
        entry_input,
        authority_context=authority_context,
        precedents=(),
        model_config=model_config,
        runtime_config=runtime_config,
        decided_by=off_label,
        provider=provider,
        on_call=on_call,
    )

    retrieved_entry = generate_llm_entry(
        entry_input,
        authority_context=authority_context,
        precedents=retrieved_precedents.precedents,
        model_config=model_config,
        runtime_config=runtime_config,
        decided_by=on_label,
        provider=provider,
        on_call=on_call,
    )

    return compose_factorial_outputs(
        bare_entry=bare_entry,
        retrieved_entry=retrieved_entry,
        resolution_context=resolution_context,
    )
