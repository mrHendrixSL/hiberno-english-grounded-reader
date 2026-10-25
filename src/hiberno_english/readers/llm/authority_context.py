"""Binding the apparatus stage's own authority tables to the exact
`authority_context` mapping `readers.llm.prompts` expects.

The canonical vocabulary column for `pos_labels`, `register_labels`, and
`language_codes` is `"term"` -- not a new choice made here, but the same
column `hiberno_english.readers.rule_based.extractor` already treats as
authoritative for these identical three tables (see its `_term_set`),
itself matching the original research constants (e.g.
`POS_TERMS = {r["term"] for r in auth("pos_labels")}`). This module
reuses that existing contract for the LLM prompt layer; it does not
invent a second one.

No file IO, no repository paths, no knowledge of E390, retrieval, model
execution, or the B/R/D/G reader conditions -- this module only
reshapes three already in-memory `pyarrow.Table` objects (as produced by
`hiberno_english.apparatus.extract_authorities`) into the plain mapping
`build_field_group_requests` reads from.
"""
from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping

import pyarrow as pa

_REQUIRED_TABLES = ("pos_labels", "register_labels", "language_codes")
_VOCABULARY_COLUMN = "term"


class LLMHasAuthorityContextError(RuntimeError):
    """Raised when the caller-supplied apparatus authority tables cannot
    be bound into the `authority_context` mapping the LLM prompt layer
    expects, on trustworthy input. No partial context is ever returned
    alongside this exception."""


def _extract_terms(table: object, name: str) -> tuple[str, ...]:
    if not isinstance(table, pa.Table):
        raise LLMHasAuthorityContextError(
            f"authority_tables[{name!r}] must be a pyarrow.Table, got "
            f"{type(table).__name__}")

    names = table.schema.names
    col_counts = Counter(names)
    dup_cols = sorted(n for n, c in col_counts.items() if c > 1)
    if dup_cols:
        raise LLMHasAuthorityContextError(
            f"authority table {name!r} has duplicate column name(s): {dup_cols}")
    if _VOCABULARY_COLUMN not in names:
        raise LLMHasAuthorityContextError(
            f"authority table {name!r} is missing the canonical vocabulary "
            f"column {_VOCABULARY_COLUMN!r}")

    terms: list[str] = []
    for i, row in enumerate(table.to_pylist()):
        value = row[_VOCABULARY_COLUMN]
        if value is None:
            raise LLMHasAuthorityContextError(
                f"authority table {name!r} row {i}: {_VOCABULARY_COLUMN!r} is null")
        if not isinstance(value, str):
            raise LLMHasAuthorityContextError(
                f"authority table {name!r} row {i}: {_VOCABULARY_COLUMN!r} must "
                f"be a string, got {type(value).__name__}")
        if not value.strip():
            raise LLMHasAuthorityContextError(
                f"authority table {name!r} row {i}: {_VOCABULARY_COLUMN!r} is "
                "empty or whitespace-only")
        # The exact printed string is kept, never stripped/normalised --
        # .strip() above is only an emptiness check, not a transform.
        terms.append(value)
    return tuple(terms)


def build_llm_authority_context(
    authority_tables: Mapping[str, pa.Table],
) -> Mapping[str, tuple[str, ...]]:
    """Build the exact `authority_context` mapping
    `hiberno_english.readers.llm.prompts.build_field_group_requests`
    expects, from the apparatus stage's own tables
    (`extract_authorities(...).tables`), consumed exactly as supplied --
    no file IO, no sorting, no normalisation, no deduplication.

    Requires `authority_tables` to contain `pos_labels`,
    `register_labels`, and `language_codes`; every row's `term` column
    value must be a non-null, non-empty (after stripping whitespace for
    the check only) string. A duplicate `term` value within one table is
    preserved exactly as printed, in table row order, never rejected or
    collapsed -- the existing authority contract
    (`hiberno_english.readers.rule_based.extractor._term_set`) does not
    treat a duplicate term as an error for these same three tables
    either, so this binding does not invent a stricter one.

    Returns one immutable mapping with exactly the three keys
    `"pos_labels"`, `"register_labels"`, `"language_codes"`, each a
    tuple of strings in the source table's own row order. Passing the
    SAME returned mapping to both the bare and retrieved generation
    batteries of one `run_factorial_entry` call is what keeps their
    authority vocabulary value-for-value identical -- this function
    performs no condition-specific branching of any kind.

    Raises `LLMHasAuthorityContextError` on a missing required table, a
    malformed table type, a missing or duplicate vocabulary column, or a
    null/non-string/empty vocabulary value -- never silently drops a
    malformed row.
    """
    missing = [t for t in _REQUIRED_TABLES if t not in authority_tables]
    if missing:
        raise LLMHasAuthorityContextError(
            f"authority_tables is missing required table(s): {missing}")

    return MappingProxyType({
        name: _extract_terms(authority_tables[name], name)
        for name in _REQUIRED_TABLES
    })
