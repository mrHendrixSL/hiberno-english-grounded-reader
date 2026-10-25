"""The final rule-based reader: composes the two already-canonical layers.

    RB_raw = extract_rule_based_entry(...)   -- unresolved lexical extraction
    RB     = rho(RB_raw)                     -- resolve_entry(RB_raw, ...)

This is the ONLY public definition of final RB. `RB_raw` remains directly
available through `extract_rule_based_entry` for auditing and
architectural decomposition, never as an alternate final condition -- RB
is always the resolved deterministic baseline.

No resolution logic lives here: headword normalisation/indexing, pointer
aliases, informant/locality matching, and bibliography lookup belong
exclusively to `hiberno_english.resolution`. This module only calls the
two existing layers in sequence and re-verifies, at this composition
boundary, the same annotation-only invariant `resolve_entry` itself
already guarantees.
"""
from __future__ import annotations

from typing import Mapping

from ...resolution import DeterministicResolutionContext, resolve_entry
from ...schema import Entry, Provenance
from .extractor import RuleBasedContext, RuleBasedReaderError, extract_rule_based_entry


def _masked_dump(entry: Entry) -> dict:
    """`entry.model_dump` with every resolution-only field reset to its
    unresolved default, for comparing RB_raw against RB while ignoring
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


def finalize_rule_based_entry(
    raw_entry: Entry,
    resolution_context: DeterministicResolutionContext,
) -> Entry:
    """RB = rho(RB_raw).

    `raw_entry` is never mutated: `resolve_entry` already operates on a
    deep copy, and that is re-verified here since this is the
    experimental boundary the RB/RB_raw distinction depends on. Raises
    `RuleBasedReaderError` if resolution changed anything besides
    `Region.resolved_name`/`.source_type` or `CrossReference.resolution`/
    `.targets`/`.resolves` -- unreachable on `resolve_entry`'s current
    implementation, but checked here anyway, the same defensive pattern
    used at the B/R/D/G factorial composition boundary.
    """
    before = raw_entry.model_copy(deep=True)
    final = resolve_entry(raw_entry, resolution_context)

    if raw_entry != before:
        raise RuleBasedReaderError(
            "finalize_rule_based_entry: raw_entry was mutated during resolution")

    if _masked_dump(raw_entry) != _masked_dump(final):
        raise RuleBasedReaderError(
            "finalize_rule_based_entry: resolution changed a field other than "
            "Region.resolved_name/.source_type or CrossReference.resolution/"
            ".targets/.resolves -- the RB_raw -> RB annotation-only invariant "
            "was violated")

    return final


def read_rule_based_entry(
    source_entry: Mapping[str, object],
    extraction_context: RuleBasedContext,
    resolution_context: DeterministicResolutionContext,
    *,
    provenance: Provenance,
) -> Entry:
    """The complete final RB reader for one source entry: RB_raw
    extraction followed immediately by rho. No other processing."""
    raw = extract_rule_based_entry(source_entry, extraction_context, provenance=provenance)
    return finalize_rule_based_entry(raw, resolution_context)
