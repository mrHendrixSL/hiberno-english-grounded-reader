"""Structural precedent retrieval: a frozen index of candidate structural
coordinates, and Euclidean-distance top-k lookup against it.

The final method is: source-side phi -> the fixed clustering/retrieval
projection phi_C -> the already-fitted, frozen structural transform T ->
PCA coordinates z -> ascending Euclidean distance -> top-k. This module
implements the index and the lookup only; phi_C selection and T itself
(log1p, StandardScaler, PCA) live in `hiberno_english.features.transform`
and are never reimplemented or refit here -- this module only ever calls
`apply_structural_transform` on an already-fitted `StructuralTransform`.

An index built here indexes STRUCTURE only: entry_id and frozen PCA
coordinates. It carries no source text, no adjudicated reading content,
and no reader output -- a later stage joins retrieved precedent_entry_ids
to the adjudicated E390 bank for anything beyond structure.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa

from ..features import StructuralTransform, StructuralTransformError, apply_structural_transform


class StructuralRetrievalError(RuntimeError):
    """Raised when a structural precedent index cannot be built or queried
    on trustworthy input, when a supplied StructuralTransform's fingerprint
    does not match the index it is being applied to, or when an internal
    invariant of retrieval itself is violated. No partial result is ever
    returned alongside this exception.
    """


def _transform_fingerprint(transform: StructuralTransform) -> str:
    """A deterministic SHA-256 fingerprint of a fitted StructuralTransform's
    complete state: `to_dict()`, canonicalised via `json.dumps(sort_keys=True,
    separators=(",", ":"), allow_nan=False)`, encoded UTF-8, then hashed.
    Two StructuralTransform instances with identical fitted state always
    fingerprint identically; this is what lets a precedent index detect
    that it is being queried with a different T than the one that built it.
    """
    try:
        canonical = json.dumps(
            transform.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StructuralRetrievalError(
            f"cannot fingerprint StructuralTransform: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_frozen_structural_transform(
    path: Path, *, expected_sha256: str | None = None,
) -> StructuralTransform:
    """Load an already-fitted `StructuralTransform` from a persisted JSON
    file (the shape `StructuralTransform.to_dict`/`from_dict` share) and
    verify its canonical fingerprint (`_transform_fingerprint`, the same
    one `StructuralPrecedentIndex.transform_sha256` uses) -- this is the
    ONLY loading path a held-out/production consumer should use for T.
    It NEVER refits, and NEVER falls back to `hiberno_english.features.
    fit_structural_transform` for a missing or mismatched file; a
    development/reproduction utility that genuinely needs to refit T
    from scratch calls that function directly, never through here.

    Raises `StructuralRetrievalError` if `path` does not exist, is not
    valid JSON, does not parse as a `StructuralTransform` (see
    `StructuralTransform.from_dict`), or -- when `expected_sha256` is
    given -- if the loaded transform's own canonical fingerprint does
    not match it exactly.
    """
    if not path.exists():
        raise StructuralRetrievalError(
            f"frozen structural transform not found at {path} -- refusing "
            "to silently refit one; this path must already exist as a "
            "persisted, frozen artefact before any held-out/production "
            "retrieval runs")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StructuralRetrievalError(f"cannot read {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StructuralRetrievalError(f"{path} is not valid JSON: {exc}") from exc

    transform = StructuralTransform.from_dict(data)

    if expected_sha256 is not None:
        actual = _transform_fingerprint(transform)
        if actual != expected_sha256:
            raise StructuralRetrievalError(
                f"{path}: loaded transform's canonical fingerprint "
                f"{actual!r} does not match the expected frozen value "
                f"{expected_sha256!r} -- refusing to use a transform whose "
                "provenance does not match; this file may have been "
                "refit, edited, or replaced since it was frozen")

    return transform


def _is_valid_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class StructuralPrecedentIndex:
    """A frozen index of one candidate bank's structural PCA coordinates.

    `entry_ids` is the candidate bank in its authoritative order (for the
    frozen E390 workflow, E390's own canonical corpus order) -- never
    reordered internally. `coordinates[i]` is entry_ids[i]'s frozen z
    vector, `z = T(phi_C(entry))`, for the T fingerprinted by
    `transform_sha256`. Retrieval refuses to apply a differently-fingerprinted
    T against this index.
    """

    entry_ids: tuple[str, ...]
    coordinates: tuple[tuple[float, ...], ...]
    transform_sha256: str

    def __post_init__(self) -> None:
        n = len(self.entry_ids)
        if n == 0:
            raise StructuralRetrievalError("precedent index cannot be empty")
        if len(set(self.entry_ids)) != n:
            raise StructuralRetrievalError("index entry_ids must be unique")
        if len(self.coordinates) != n:
            raise StructuralRetrievalError(
                "index coordinate row count must equal entry_id count: "
                f"{len(self.coordinates)} != {n}")
        dim = len(self.coordinates[0])
        if dim < 1:
            raise StructuralRetrievalError(
                "index coordinate dimensionality must be >= 1")
        for row in self.coordinates:
            if len(row) != dim:
                raise StructuralRetrievalError(
                    "every index coordinate row must have the same "
                    f"dimensionality ({dim})")
            for v in row:
                if not math.isfinite(v):
                    raise StructuralRetrievalError(
                        f"index coordinates must be finite, got {v!r}")
        if not _is_valid_sha256_hex(self.transform_sha256):
            raise StructuralRetrievalError(
                "transform_sha256 must be a 64-character lowercase hex "
                f"SHA-256 digest, got {self.transform_sha256!r}")

    def to_dict(self) -> dict:
        """A plain JSON-compatible representation -- strings, ints, floats,
        lists only. Never a pickle, and never the StructuralTransform
        itself: the transform remains a separately frozen object,
        identified here only by its fingerprint."""
        return {
            "entry_ids": list(self.entry_ids),
            "coordinates": [list(row) for row in self.coordinates],
            "transform_sha256": self.transform_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuralPrecedentIndex":
        """Reconstruct from `to_dict()`'s output. Raises
        StructuralRetrievalError on a missing key or on any shape that
        fails the same checks the constructor enforces on every index."""
        try:
            return cls(
                entry_ids=tuple(data["entry_ids"]),
                coordinates=tuple(
                    tuple(float(v) for v in row) for row in data["coordinates"]),
                transform_sha256=data["transform_sha256"],
            )
        except KeyError as exc:
            raise StructuralRetrievalError(
                f"malformed serialised index: missing key {exc}") from exc


def build_structural_precedent_index(
    features: pa.Table,
    precedent_entry_ids: Sequence[str],
    transform: StructuralTransform,
) -> StructuralPrecedentIndex:
    """Build a frozen structural-coordinate index over `precedent_entry_ids`
    (for the frozen E390 workflow: `design_e390(...).selected_entry_ids`),
    using the supplied, already-fitted `transform`. Fits nothing.

    `features` may be the full corpus feature table; only the rows named
    by `precedent_entry_ids` are selected, in the order supplied -- this
    function never sorts or otherwise reorders the candidate bank. Those
    rows are projected through `apply_structural_transform`; phi_C
    selection, log1p, scaling, and PCA all happen there, not here.

    Raises StructuralRetrievalError if `precedent_entry_ids` is empty,
    contains a duplicate, names an entry_id absent from `features`, or
    names an entry_id that appears more than once in `features` (so no
    single row could be selected unambiguously).
    """
    ids = list(precedent_entry_ids)
    if not ids:
        raise StructuralRetrievalError("precedent bank cannot be empty")

    id_counts = Counter(ids)
    dup_ids = sorted(i for i, n in id_counts.items() if n > 1)
    if dup_ids:
        raise StructuralRetrievalError(
            f"duplicate precedent_entry_id(s): {dup_ids}")

    if "entry_id" not in features.schema.names:
        raise StructuralRetrievalError("feature table has no 'entry_id' column")

    rows_by_id: dict[str, list[dict]] = defaultdict(list)
    for r in features.to_pylist():
        rows_by_id[r["entry_id"]].append(r)

    missing_ids = [i for i in ids if len(rows_by_id.get(i, [])) == 0]
    if missing_ids:
        raise StructuralRetrievalError(
            f"precedent_entry_id(s) absent from the feature table: {missing_ids}")
    ambiguous_ids = sorted({i for i in ids if len(rows_by_id.get(i, [])) > 1})
    if ambiguous_ids:
        raise StructuralRetrievalError(
            "precedent_entry_id(s) appear more than once in the feature "
            f"table, cannot select an unambiguous row: {ambiguous_ids}")

    ordered_rows = [rows_by_id[i][0] for i in ids]
    precedent_table = pa.Table.from_pylist(ordered_rows)

    try:
        z_table = apply_structural_transform(transform, precedent_table)
    except StructuralTransformError as exc:
        raise StructuralRetrievalError(
            f"cannot transform precedent features to build the index: {exc}"
        ) from exc

    z_columns = [f"z_{j}" for j in range(transform.n_components)]
    coords_matrix = np.column_stack(
        [z_table.column(c).to_numpy() for c in z_columns]).astype(np.float64)
    coordinates = tuple(tuple(float(v) for v in row) for row in coords_matrix)

    return StructuralPrecedentIndex(
        entry_ids=tuple(ids),
        coordinates=coordinates,
        transform_sha256=_transform_fingerprint(transform),
    )


def retrieve_structural_precedents(
    index: StructuralPrecedentIndex,
    transform: StructuralTransform,
    query_features: pa.Table,
    *,
    k: int = 3,
) -> pa.Table:
    """Retrieve, for every row of `query_features`, the top-k structural
    precedents from `index` by ascending Euclidean distance in the frozen
    PCA coordinate space: d(z_q, z_e) = ||z_q - z_e||_2, z = T(phi_C(.)).
    z vectors are never normalised or whitened; the reported distance is
    the actual Euclidean distance, never a similarity score or its square.

    The candidate pool for a query already in the index is the index minus
    that one entry_id (self-exclusion is by entry_id, never by coordinate
    equality: two distinct entries with identical coordinates remain
    legitimate candidates for each other). A query not present in the
    index excludes nothing. Ties are broken deterministically: ascending
    distance first, then ascending original candidate-bank index -- never
    by content, score, or random choice.

    Returns a table of `query_entry_id`, `rank` (1-based), `precedent_entry_id`,
    `distance`, ordered by input query order and then by rank. Up to `k`
    rows per query; fewer only when fewer than `k` candidates remain after
    self-exclusion -- never an error for that reason alone.

    Raises StructuralRetrievalError if `k` is not positive, if `transform`'s
    fingerprint does not match `index.transform_sha256` (a query must never
    be scored against an index built under a different T), if the
    transformed query dimensionality does not match the index's, if
    `query_features` fails phi_C/T's own input contract (wrapped from the
    underlying StructuralTransformError), or if a computed distance is
    non-finite.
    """
    if k <= 0:
        raise StructuralRetrievalError(f"k must be positive, got {k}")

    supplied_fingerprint = _transform_fingerprint(transform)
    if supplied_fingerprint != index.transform_sha256:
        raise StructuralRetrievalError(
            "transform fingerprint mismatch: the supplied StructuralTransform "
            "is not the one used to build this index -- refusing to score "
            f"query coordinates against it (supplied={supplied_fingerprint}, "
            f"index={index.transform_sha256})")

    try:
        z_table = apply_structural_transform(transform, query_features)
    except StructuralTransformError as exc:
        raise StructuralRetrievalError(
            f"cannot transform query features for retrieval: {exc}") from exc

    index_dim = len(index.coordinates[0])
    if transform.n_components != index_dim:
        raise StructuralRetrievalError(
            "transformed query dimensionality does not match the index: "
            f"query={transform.n_components}, index={index_dim}")

    z_columns = [f"z_{j}" for j in range(transform.n_components)]
    query_ids = z_table.column("entry_id").to_pylist()
    query_matrix = np.column_stack(
        [z_table.column(c).to_numpy() for c in z_columns]).astype(np.float64)

    bank_matrix = np.array(index.coordinates, dtype=np.float64)
    bank_ids = index.entry_ids

    out_rows: list[dict] = []
    for qi, qid in enumerate(query_ids):
        distances = np.linalg.norm(bank_matrix - query_matrix[qi], axis=1)
        if not np.all(np.isfinite(distances)):
            bad = [bank_ids[j] for j in range(len(bank_ids))
                   if not math.isfinite(distances[j])]
            raise StructuralRetrievalError(
                f"non-finite distance computed for query {qid!r} against "
                f"precedent(s) {bad}")

        candidates = [
            (float(distances[j]), j) for j in range(len(bank_ids)) if bank_ids[j] != qid]
        candidates.sort(key=lambda t: (t[0], t[1]))

        for rank, (dist, j) in enumerate(candidates[:k], start=1):
            out_rows.append({
                "query_entry_id": qid,
                "rank": rank,
                "precedent_entry_id": bank_ids[j],
                "distance": dist,
            })

    if out_rows:
        return pa.Table.from_pylist(out_rows)
    return pa.Table.from_pydict({
        "query_entry_id": pa.array([], type=pa.string()),
        "rank": pa.array([], type=pa.int64()),
        "precedent_entry_id": pa.array([], type=pa.string()),
        "distance": pa.array([], type=pa.float64()),
    })
