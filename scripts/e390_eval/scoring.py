"""Deterministic precision/recall/F1 scoring, plus the five B/R/D/G
factorial contrasts. No fuzzy matching, no embeddings, no LLM-as-judge --
every function here is a pure, side-effect-free computation over already-
extracted leaf-value lists (see `contract.extract_path`).

Aggregation is MICRO (true positives, predicted-count, and reference-count
are summed across every entry first; precision/recall/F1 are computed once
from those sums) rather than averaging per-entry ratios -- this is what
lets an entry with an empty reference or an empty prediction contribute
its counts safely without its own ratio ever needing to be computed, and
is the standard convention for slot-filling-style extraction metrics.
Per-entry ratios are also exposed (`score_multiset`) for diagnostics, using
the same None-safe convention.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

from .normalize import apply_profile


def _hashable(v: object) -> Hashable:
    """Leaf values are always JSON-scalar (str, int, float, bool, None) in
    this schema -- never a dict or list -- so they are already hashable.
    Raise loudly rather than silently coerce if that assumption is ever
    violated by a future schema change."""
    if isinstance(v, (dict, list)):
        raise TypeError(
            f"scoring expects flattened leaf values, got a {type(v).__name__}: "
            f"{v!r} -- extract_path was given a path that does not reach a leaf")
    return v


@dataclass(frozen=True)
class Counts:
    """Raw multiset-intersection counts. Never a ratio -- ratios are
    derived from these via `precision_recall_f1`, which is where the
    zero-denominator convention lives, once, in one place."""
    true_positive: int
    predicted_count: int
    reference_count: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.true_positive + other.true_positive,
            self.predicted_count + other.predicted_count,
            self.reference_count + other.reference_count,
        )


ZERO_COUNTS = Counts(0, 0, 0)


def multiset_counts(
    predicted: Sequence[object],
    reference: Sequence[object],
    *,
    normalize: Callable[[str], str] | None = None,
) -> Counts:
    """TP/predicted/reference counts for one entry's flattened leaf-value
    multiset at one field path. Order-irrelevant; duplicate values are
    preserved as duplicates (Counter multiset intersection), so a
    reference with two identical attestations requires two matching
    predicted attestations for full credit, never just one.

    `normalize`, when given, is applied to every STRING leaf on both
    sides before comparison (see `normalize.apply_profile`) -- every
    other value type is untouched. Omitted (the default), this is
    byte-for-byte the original exact-equality behaviour; every existing
    caller that never passes it is completely unaffected.
    """
    predicted = apply_profile(predicted, normalize)
    reference = apply_profile(reference, normalize)
    pred_counter = Counter(_hashable(v) for v in predicted)
    ref_counter = Counter(_hashable(v) for v in reference)
    tp = sum((pred_counter & ref_counter).values())
    return Counts(tp, sum(pred_counter.values()), sum(ref_counter.values()))


def ordered_counts(predicted: Sequence[object], reference: Sequence[object]) -> Counts:
    """Position-sensitive counterpart to `multiset_counts`, for a
    hypothetical ordered structure: true_positive counts only positions
    where predicted[i] == reference[i] for i < min(len(predicted),
    len(reference)) -- a value present in both sequences but at different
    positions is NOT a match. No field in the current public schema uses
    this (see contract.py's `FLATTENED_MULTISET_RATIONALE`); kept as a
    tested, available primitive for a future field where the source
    material asserts a meaningful print order.
    """
    tp = sum(
        1 for p, r in zip(predicted, reference)
        if _hashable(p) == _hashable(r))
    return Counts(tp, len(predicted), len(reference))


def precision_recall_f1(counts: Counts) -> dict[str, float | None]:
    """`None` (never a crash, never a silently-assumed 0 or 1) exactly
    when the corresponding denominator is 0: precision is undefined with
    no predictions, recall is undefined with no reference content. F1 is
    `None` unless both precision and recall are defined and not both
    zero."""
    precision = (counts.true_positive / counts.predicted_count
                 if counts.predicted_count > 0 else None)
    recall = (counts.true_positive / counts.reference_count
             if counts.reference_count > 0 else None)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": counts.true_positive,
            "predicted_count": counts.predicted_count,
            "reference_count": counts.reference_count}


def aggregate_counts(per_entry: Sequence[Counts]) -> Counts:
    total = ZERO_COUNTS
    for c in per_entry:
        total = total + c
    return total


def rollup_from_field_metrics(field_metrics: Mapping[str, Mapping[str, object]], paths: Sequence[str]) -> dict[str, float | None]:
    """A first-class rollup across several field paths' ALREADY-COMPUTED
    per-path metrics (as `precision_recall_f1` returns them): sums the
    raw `true_positive`/`predicted_count`/`reference_count` counts across
    `paths` first, then calls `precision_recall_f1` exactly once on the
    sum -- never an average of per-path F1 values.

    Safe to call repeatedly with different `paths` subsets against the
    same already-computed `field_metrics` mapping (e.g. a "targeted"
    subset and the "full schema" superset) without re-walking any entry
    or re-extracting anything, since summing already-micro-aggregated
    per-path counts is exactly equivalent to summing raw per-(entry,path)
    counts directly (integer addition is associative/commutative).
    """
    missing = [p for p in paths if p not in field_metrics]
    if missing:
        raise KeyError(f"rollup_from_field_metrics: unknown field path(s): {missing}")
    total = ZERO_COUNTS
    for p in paths:
        m = field_metrics[p]
        total = total + Counts(m["true_positive"], m["predicted_count"], m["reference_count"])
    return precision_recall_f1(total)


# ── B/R/D/G factorial contrasts ─────────────────────────────────────────────
#
# Every contrast is a plain subtraction of one already-computed metric
# value between two conditions of the SAME model/field/path -- never a
# recomputation, never an interpretation. `None` propagates: if either
# side of a subtraction is undefined (see `precision_recall_f1`), the
# contrast is undefined too, never silently treated as 0.
CONTRAST_DEFINITIONS: Mapping[str, tuple[str, str]] = {
    "retrieval_effect_resolver_off": ("R", "B"),   # R - B
    "retrieval_effect_resolver_on": ("G", "D"),    # G - D
    "resolution_effect_retrieval_off": ("D", "B"), # D - B
    "resolution_effect_retrieval_on": ("G", "R"),  # G - R
}


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def compute_contrasts(metrics_by_condition: Mapping[str, float | None]) -> dict[str, float | None]:
    """`metrics_by_condition` maps "B"/"R"/"D"/"G" to one already-computed
    scalar metric value (a precision, a recall, an F1, or a structural
    count) for one model, one field/path. Returns the four base contrasts
    plus the interaction term, computed two equivalent ways as a built-in
    consistency check.
    """
    missing = [c for c in ("B", "R", "D", "G") if c not in metrics_by_condition]
    if missing:
        raise KeyError(f"compute_contrasts requires B/R/D/G, missing: {missing}")

    contrasts = {
        name: _sub(metrics_by_condition[a], metrics_by_condition[b])
        for name, (a, b) in CONTRAST_DEFINITIONS.items()
    }

    interaction_via_retrieval = _sub(
        contrasts["retrieval_effect_resolver_on"], contrasts["retrieval_effect_resolver_off"])
    interaction_via_resolution = _sub(
        contrasts["resolution_effect_retrieval_on"], contrasts["resolution_effect_retrieval_off"])

    contrasts["interaction"] = interaction_via_retrieval
    contrasts["interaction_cross_check"] = interaction_via_resolution
    return contrasts
