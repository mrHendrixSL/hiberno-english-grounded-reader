"""Binding retrieved structural-precedent IDs to already-adjudicated E390
Entry objects.

This module is the narrow bridge between two layers that must otherwise
stay ignorant of each other:

    hiberno_english.retrieval.structural
        knows structure, entry_id, rank, distance -- no adjudicated
        lexical content.

    hiberno_english.readers.llm.precedents (this module)
        binds retrieved precedent_entry_ids to caller-supplied,
        already-adjudicated Entry objects -- no retrieval mathematics.

    hiberno_english.readers.llm.prompts
        formats supplied Entry precedents into prompt text -- no
        retrieval IDs, ranks, or distances.

Distance stops at this module's boundary: `QueryPrecedents` carries no
distance or rank field, and neither does any Entry it wraps. Only a
precedent's position in the output tuple encodes its retrieval rank.

The adjudicated E390 content itself remains external research/release
data. This module never reads or embeds the 390 adjudicated readings
itself -- it operates only on `Entry` objects a caller already loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import pyarrow as pa

from ...schema import Entry

REQUIRED_RETRIEVAL_COLUMNS = (
    "query_entry_id", "rank", "precedent_entry_id", "distance")


class PrecedentBindingError(RuntimeError):
    """Raised when an adjudicated precedent bank cannot be built or
    verified, or when a retrieval table cannot be bound to one, on
    trustworthy input. No partial bank or binding result is ever returned
    alongside this exception."""


def _membership_sha256(entry_ids: Sequence[str]) -> str:
    """The same ordered-membership convention used by E390 design itself
    (`E390DesignResult.ordered_membership_sha256`): SHA-256 of the
    newline-joined entry_ids, in bank order."""
    return hashlib.sha256(
        "\n".join(entry_ids).encode("utf-8")).hexdigest()


def _content_sha256(entries: Sequence[Entry]) -> str:
    """Fingerprints the complete adjudicated Entry objects, in bank order
    -- lexical fields and resolution fields both included, nothing
    excluded or stripped. This is an archival fingerprint of WHICH
    adjudicated readings were supplied, distinct from the group-specific,
    resolution-stripped view `prompts.py` later exposes to a model."""
    sequence = [e.model_dump(mode="json", exclude_none=False) for e in entries]
    canonical = json.dumps(
        sequence, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdjudicatedPrecedentBank:
    """A caller-supplied set of already-adjudicated Entry objects, in a
    fixed order, with two independent fingerprints.

    `membership_sha256` says WHICH entries comprise the bank (identical
    convention to E390 design's own membership fingerprint, so the
    historical E390 bank's value can be compared directly against the
    frozen design's own hash). `content_sha256` says WHICH adjudicated
    readings were supplied for those entries.

    Contains structured adjudicated readings only: no source text, no
    retrieval distance/rank, no feature vectors, no raw model output, no
    disagreement values. The actual E390 data remain external
    research/release data -- this bank is never committed to GitHub with
    real entries populated.
    """

    entry_ids: tuple[str, ...]
    entries: tuple[Entry, ...]
    membership_sha256: str
    content_sha256: str


def build_adjudicated_precedent_bank(
    entries: Sequence[Entry],
    *,
    expected_entry_ids: Sequence[str] | None = None,
) -> AdjudicatedPrecedentBank:
    """Build an `AdjudicatedPrecedentBank` from caller-supplied Entry
    objects, preserving caller order exactly -- never sorted.

    Raises `PrecedentBindingError` if `entries` is empty, if any item is
    not a `hiberno_english.schema.Entry`, if any `Entry.entry_id` is
    empty, if entry_ids are not unique, if `expected_entry_ids` contains a
    duplicate, or if `entry_ids` does not exactly equal
    `expected_entry_ids` (same members, same order) when supplied.

    For the frozen E390 workflow, pass
    `expected_entry_ids=design_e390(...).selected_entry_ids` so bank
    membership and order are explicitly tied to the frozen E390 design
    rather than merely assumed.
    """
    entries = tuple(entries)
    if not entries:
        raise PrecedentBindingError("adjudicated precedent bank cannot be empty")

    for i, e in enumerate(entries):
        if not isinstance(e, Entry):
            raise PrecedentBindingError(
                f"bank item {i} is not a hiberno_english.schema.Entry, "
                f"got {type(e).__name__}")
        if not e.entry_id:
            raise PrecedentBindingError(f"bank item {i} has an empty entry_id")

    entry_ids = tuple(e.entry_id for e in entries)
    counts = Counter(entry_ids)
    dup_ids = sorted(i for i, n in counts.items() if n > 1)
    if dup_ids:
        raise PrecedentBindingError(f"duplicate entry_id(s) in bank: {dup_ids}")

    if expected_entry_ids is not None:
        expected = tuple(expected_entry_ids)
        exp_counts = Counter(expected)
        exp_dups = sorted(i for i, n in exp_counts.items() if n > 1)
        if exp_dups:
            raise PrecedentBindingError(
                f"duplicate id(s) in expected_entry_ids: {exp_dups}")
        if entry_ids != expected:
            raise PrecedentBindingError(
                "bank entry_ids do not exactly match expected_entry_ids "
                f"(order and membership both matter): got {len(entry_ids)} "
                f"entries, expected {len(expected)}")

    return AdjudicatedPrecedentBank(
        entry_ids=entry_ids,
        entries=entries,
        membership_sha256=_membership_sha256(entry_ids),
        content_sha256=_content_sha256(entries),
    )


def _verify_bank_integrity(bank: AdjudicatedPrecedentBank) -> None:
    """A frozen dataclass does not make its nested Pydantic Entry objects
    themselves immutable, so a caller could mutate one after bank
    construction. Recompute entry_ids/membership_sha256/content_sha256
    from `bank.entries` and compare against the bank's stored state
    before any binding uses it; raise rather than repair on mismatch."""
    recomputed_ids = tuple(e.entry_id for e in bank.entries)
    if recomputed_ids != bank.entry_ids:
        raise PrecedentBindingError(
            "adjudicated bank entry_ids no longer match its entries -- a "
            "bank Entry's entry_id was mutated after construction")

    if _membership_sha256(recomputed_ids) != bank.membership_sha256:
        raise PrecedentBindingError(
            "adjudicated bank membership fingerprint no longer matches "
            "its entries -- the bank was mutated after construction")

    if _content_sha256(bank.entries) != bank.content_sha256:
        raise PrecedentBindingError(
            "adjudicated bank content fingerprint no longer matches its "
            "entries -- an adjudicated Entry was mutated after bank "
            "construction")


@dataclass(frozen=True)
class QueryPrecedents:
    """One query's ordered adjudicated precedent set: suitable to pass
    directly as `precedents=query.precedents` to
    `build_field_group_requests()`.

    Carries no retrieval distance, rank, or similarity score -- a
    precedent's position in the tuple is what encodes its retrieval rank.
    `precedent_entry_ids` and `precedents` always have identical length
    and positional correspondence.
    """

    query_entry_id: str
    precedent_entry_ids: tuple[str, ...]
    precedents: tuple[Entry, ...]

    def __post_init__(self) -> None:
        if len(self.precedent_entry_ids) != len(self.precedents):
            raise PrecedentBindingError(
                "QueryPrecedents.precedent_entry_ids and .precedents must "
                "have identical length: "
                f"{len(self.precedent_entry_ids)} != {len(self.precedents)}")
        for i, (pid, entry) in enumerate(zip(self.precedent_entry_ids, self.precedents)):
            if entry.entry_id != pid:
                raise PrecedentBindingError(
                    f"QueryPrecedents position {i}: precedent_entry_ids "
                    f"{pid!r} does not match precedents[{i}].entry_id "
                    f"{entry.entry_id!r}")


def _validate_retrieval_schema(retrieval: pa.Table) -> None:
    names = retrieval.schema.names
    if len(set(names)) != len(names):
        col_counts = Counter(names)
        dup_cols = sorted(n for n, c in col_counts.items() if c > 1)
        raise PrecedentBindingError(
            f"retrieval table has duplicate column name(s): {dup_cols}")
    missing = [c for c in REQUIRED_RETRIEVAL_COLUMNS if c not in names]
    if missing:
        raise PrecedentBindingError(
            f"retrieval table is missing required column(s): {missing}")


def _validate_retrieval_rows(rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        qid, pid, rank, dist = (
            r["query_entry_id"], r["precedent_entry_id"], r["rank"], r["distance"])
        if not isinstance(qid, str) or not qid:
            raise PrecedentBindingError(
                f"retrieval row {i}: query_entry_id must be a non-empty "
                f"string, got {qid!r}")
        if not isinstance(pid, str) or not pid:
            raise PrecedentBindingError(
                f"retrieval row {i}: precedent_entry_id must be a "
                f"non-empty string, got {pid!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise PrecedentBindingError(
                f"retrieval row {i}: rank must be a positive integer, "
                f"got {rank!r}")
        if (isinstance(dist, bool) or not isinstance(dist, (int, float))
               or not math.isfinite(dist) or dist < 0):
            raise PrecedentBindingError(
                f"retrieval row {i}: distance must be finite and >= 0, "
                f"got {dist!r}")


def bind_retrieved_precedents(
    retrieval: pa.Table,
    bank: AdjudicatedPrecedentBank,
) -> tuple[QueryPrecedents, ...]:
    """Bind a structural-retrieval result table to an adjudicated
    precedent bank, returning one `QueryPrecedents` per query, in the
    query order the retrieval table presents.

    `retrieval` must carry exactly the columns `query_entry_id`, `rank`,
    `precedent_entry_id`, `distance` (rejecting a duplicate column name
    outright). This function validates that table -- non-empty string
    IDs, a positive integer rank, a finite non-negative distance -- but
    never recomputes retrieval itself.

    For every query block (rows sharing one `query_entry_id`, required to
    be contiguous -- the same query reappearing in a second, separate
    block is rejected): ranks must be exactly `1..n` with no gaps or
    duplicates, precedent IDs must be unique within the query, a query
    must never precede itself, and every precedent ID must exist in
    `bank`. Rank is authoritative for output order -- precedents are
    placed in ascending-rank order regardless of the table's row order;
    no other sort (by entry_id, lexical content, bank order, or distance)
    is ever applied.

    Distance is read only to validate it and is then discarded: it never
    appears in the returned `QueryPrecedents`, nor in any Entry object
    they carry. `bank` integrity is re-verified before binding (see
    `_verify_bank_integrity`), so a mutation of a bank Entry after
    construction is caught here rather than silently propagating.

    An empty `retrieval` table (correct schema, zero rows) returns `()`:
    there are no query blocks to bind, and no query identity is invented.
    """
    _verify_bank_integrity(bank)
    _validate_retrieval_schema(retrieval)

    rows = retrieval.select(list(REQUIRED_RETRIEVAL_COLUMNS)).to_pylist()
    _validate_retrieval_rows(rows)

    if not rows:
        return ()

    lookup: dict[str, Entry] = dict(zip(bank.entry_ids, bank.entries))

    blocks: list[list[dict]] = []
    seen_queries: set[str] = set()
    for r in rows:
        qid = r["query_entry_id"]
        if blocks and blocks[-1][0]["query_entry_id"] == qid:
            blocks[-1].append(r)
        else:
            if qid in seen_queries:
                raise PrecedentBindingError(
                    f"query_entry_id {qid!r} appears in more than one "
                    "non-contiguous block of the retrieval table")
            seen_queries.add(qid)
            blocks.append([r])

    results: list[QueryPrecedents] = []
    for block in blocks:
        qid = block[0]["query_entry_id"]
        n = len(block)
        block_ranks = sorted(r["rank"] for r in block)
        if block_ranks != list(range(1, n + 1)):
            raise PrecedentBindingError(
                f"query {qid!r}: ranks must be exactly 1..{n} with no "
                f"gaps or duplicates, got {sorted(r['rank'] for r in block)}")

        ordered = sorted(block, key=lambda r: r["rank"])
        pids = [r["precedent_entry_id"] for r in ordered]

        pid_counts = Counter(pids)
        dup_pids = sorted(p for p, c in pid_counts.items() if c > 1)
        if dup_pids:
            raise PrecedentBindingError(
                f"query {qid!r}: duplicate precedent_entry_id(s) within "
                f"the query: {dup_pids}")

        if qid in pids:
            raise PrecedentBindingError(
                f"query {qid!r}: precedent list contains the query's own "
                "entry_id")

        missing = [p for p in pids if p not in lookup]
        if missing:
            raise PrecedentBindingError(
                f"query {qid!r}: precedent_entry_id(s) absent from the "
                f"adjudicated bank: {missing}")

        results.append(QueryPrecedents(
            query_entry_id=qid,
            precedent_entry_ids=tuple(pids),
            precedents=tuple(lookup[p] for p in pids),
        ))

    return tuple(results)
