"""Deterministic rule-based lexical extraction and its final reader.

    RB_raw = extract_rule_based_entry(...)   -- unresolved lexical extraction
    RB     = rho(RB_raw)                     -- read_rule_based_entry(...)

`RB` (via `read_rule_based_entry`/`finalize_rule_based_entry`) is the only
public definition of final RB; `extract_rule_based_entry` remains
available directly for the unresolved `RB_raw`, for auditing and
architectural decomposition, never as an alternate final condition. No
resolution logic lives in this package -- see
`hiberno_english.resolution.resolve_entry`.
"""
from .extractor import (
    RuleBasedContext,
    RuleBasedReaderError,
    build_rule_based_context,
    extract_rule_based_entry,
)
from .reader import finalize_rule_based_entry, read_rule_based_entry

__all__ = [
    "RuleBasedContext",
    "RuleBasedReaderError",
    "build_rule_based_context",
    "extract_rule_based_entry",
    "finalize_rule_based_entry",
    "read_rule_based_entry",
]
