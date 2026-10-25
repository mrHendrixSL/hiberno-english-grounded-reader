"""The E390 stratified probability sample design: hard strata, GMM bulk
clustering, Neyman allocation, and the frozen draw.

This module reproduces the EXACT executable procedure the manuscript locks
(Sec. 4.2): three protected hard strata checked in a fixed precedence
(multi-sense > homonym > pointer), a Gaussian-mixture partition of
everything else fit over the canonical structural transform T, Neyman
allocation of the residual budget across the homonym stratum and every
cluster stratum, and a seeded draw. Given the SAME feature population and
the SAME frozen archival disagreement vector this design was originally
computed against, it reproduces the existing, already-adjudicated E390
membership exactly -- it does not draw a new sample.

This module owns stratification, allocation, and the draw ONLY:

- phi_C/T (log1p -> StandardScaler -> PCA) are never reimplemented here --
  see `hiberno_english.features.transform`. This module only calls
  `fit_structural_transform`/`apply_structural_transform`.
- disagreement is treated as a frozen, external, archival design input,
  supplied by the caller as a plain (entry_id, disagreement) table. This
  module never reads a reader's extraction output, never knows a model's
  name, and never computes pairwise reader disagreement itself. The
  historical disagreement vector was computed once, at the 2026-08-21
  design snapshot, from three specific historical readers; none of that
  is a dependency of this module, only of the archival table it is handed.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pyarrow as pa
from sklearn.mixture import GaussianMixture

from ..features import (
    CLUSTER_FEATURES,
    StructuralTransform,
    apply_structural_transform,
    fit_structural_transform,
)

# Minimum per stratum: below about 10 the within-stratum variance cannot be
# estimated from the sample itself.
MIN_PER_STRATUM = 10
# Smallest admissible GMM component. The largest k whose smallest fitted
# component still holds this many entries is selected -- not the BIC
# minimum; BIC is computed per candidate as a diagnostic only.
MIN_CLUSTER_N = 100
# Pointer entries ("X, see Y.") are structurally trivial: capped, not
# Neyman-allocated.
POINTER_CAP = 20

_HARD_STRATUM_COLUMNS = ("n_sense_num", "has_homonym", "is_pointer", "is_cross_ref")
_REQUIRED_FEATURE_COLUMNS = (
    ("entry_id", "homonym_index", "para_idx")
    + _HARD_STRATUM_COLUMNS
    + CLUSTER_FEATURES
)


class E390DesignError(RuntimeError):
    """Raised when the E390 design cannot be computed on trustworthy input,
    or when an internal invariant of the design procedure itself is
    violated. No partial or silently-incomplete design is ever returned
    alongside this exception.
    """


@dataclass(frozen=True)
class GMMCandidateDiagnostic:
    """One candidate component count considered while selecting the bulk
    GMM's k. `bic` is recorded for inspection only -- it never selects k;
    `selected` marks whichever candidate the design criterion (the largest
    k whose smallest fitted component holds >= MIN_CLUSTER_N entries)
    actually chose.
    """

    k: int
    bic: float
    smallest_component_n: int
    selected: bool


@dataclass(frozen=True)
class E390DesignReport:
    """Summary diagnostics for one design_e390() run."""

    seed: int
    budget: int
    frame_size: int
    bulk_size: int
    selected_k: int
    retained_pca_components: int
    explained_variance: float
    gmm_candidates: tuple[GMMCandidateDiagnostic, ...]
    nominal_allocation: MappingProxyType
    realised_allocation: MappingProxyType
    population_by_stratum: MappingProxyType
    realised_selected_size: int


@dataclass(frozen=True)
class E390DesignResult:
    """The complete, inspectable outcome of one design_e390() run.

    `selected_entry_ids` is ordered exactly as the input `features` table
    was ordered (canonical corpus/feature order), never as an unordered
    set. `design_table` carries one row per input entry (stratum, homonym
    group, selection, population/allocation figures): the public design
    table contains only fields used by the final method, not source
    prose or reader output.
    """

    selected_entry_ids: tuple[str, ...]
    transform: StructuralTransform
    report: E390DesignReport
    design_table: pa.Table

    def ordered_membership_sha256(self) -> str:
        """The frozen acceptance fingerprint: sha256 of the newline-joined,
        canonically-ordered selected entry_id sequence."""
        return hashlib.sha256(
            "\n".join(self.selected_entry_ids).encode("utf-8")
        ).hexdigest()


def _hard_stratum(row: dict) -> str | None:
    """First-match precedence, executable specification: multi-sense >
    homonym > pointer > (None, i.e. bulk)."""
    if row["n_sense_num"] > 0:
        return "H_multisense"
    if row["has_homonym"]:
        return "H_homonym"
    if row["is_pointer"] or row["is_cross_ref"]:
        return "H_pointer"
    return None


def _homonym_groups(hom_rows: list[dict]) -> dict[str, str]:
    """entry_id -> group id, for every H_homonym row.

    A group is a maximal run of consecutive-by-para_idx rows whose
    homonym_index ascends from 1. Group id is `H` + the first member's
    entry_id (first in para_idx order).

    Raises E390DesignError if any H_homonym entry ends up with no group,
    in more than one group, or if a generated group is empty -- none of
    these can occur given the algorithm's own construction and
    duplicate-entry_id already being rejected upstream, but the guarantee
    is enforced explicitly rather than merely assumed.
    """
    hom = sorted(hom_rows, key=lambda f: f["para_idx"])
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for f in hom:
        if cur and (
            f["homonym_index"] == 1
            or f["homonym_index"] != cur[-1]["homonym_index"] + 1
            or f["para_idx"] != cur[-1]["para_idx"] + 1
        ):
            groups.append(cur)
            cur = [f]
        else:
            cur.append(f)
    if cur:
        groups.append(cur)

    out: dict[str, str] = {}
    for g in groups:
        if not g:
            raise E390DesignError(
                "internal error: homonym grouping produced an empty group")
        gid = f"H{g[0]['entry_id']}"
        for f in g:
            if f["entry_id"] in out:
                raise E390DesignError(
                    f"entry {f['entry_id']!r} was assigned to more than "
                    "one homonym group")
            out[f["entry_id"]] = gid

    missing = [f["entry_id"] for f in hom_rows if f["entry_id"] not in out]
    if missing:
        raise E390DesignError(
            f"H_homonym entries with no assigned homonym group: {missing}")
    return out


def _cluster_bulk(
    Z: np.ndarray, seed: int
) -> tuple[np.ndarray, int, tuple[GMMCandidateDiagnostic, ...]]:
    """Gaussian mixture over the (already-transformed) bulk coordinates Z.

    Preserves sklearn's own component labels exactly -- never reordered or
    relabelled by centroid, size, or BIC. k is chosen as the largest
    candidate in {2, ..., 10} whose smallest fitted component holds
    >= MIN_CLUSTER_N entries; if no candidate qualifies, falls back to
    k=2 (same covariance/seed/n_init). BIC is computed and recorded per
    candidate but never used to select k.
    """
    records: list[tuple[int, float, int]] = []
    chosen: tuple[int, np.ndarray] | None = None
    for k in range(2, 11):
        gm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=seed, n_init=3)
        gm.fit(Z)
        labels = gm.predict(Z)
        smallest = int(np.bincount(labels, minlength=k).min())
        records.append((k, float(gm.bic(Z)), smallest))
        if smallest >= MIN_CLUSTER_N:
            chosen = (k, labels)
    if chosen is None:
        gm = GaussianMixture(
            n_components=2, covariance_type="full", random_state=seed, n_init=3
        ).fit(Z)
        chosen = (2, gm.predict(Z))
    k, labels = chosen
    diagnostics = tuple(
        GMMCandidateDiagnostic(k=kk, bic=bb, smallest_component_n=sm, selected=(kk == k))
        for kk, bb, sm in records)
    return labels, k, diagnostics


def _neyman(
    strata: dict[str, list[dict]], budget: int, fixed: dict[str, int]
) -> dict[str, int]:
    """n_h proportional to N_h * S_h over the non-fixed strata, with the
    min-floor/population-cap repaired by iterative re-proportioning and
    residual rounding drift assigned to the largest non-fixed stratum by
    population. Returns NOMINAL allocations only; the draw may realise a
    different size for H_homonym (see module docstring / E390DesignReport).

    Raises E390DesignError if every stratum is fixed and the budget
    exceeds the fixed allocations' total -- there is then no non-fixed
    stratum left to absorb the residual budget.
    """
    alloc: dict[str, int] = dict(fixed)
    remaining = budget - sum(fixed.values())
    free: dict[str, list[dict]] = {h: rows for h, rows in strata.items() if h not in fixed}

    stats: dict[str, tuple[int, float]] = {}
    for h, rows in free.items():
        d = np.array([r["disagreement"] for r in rows], dtype=float)
        d = d[~np.isnan(d)]
        stats[h] = (len(rows), max(float(d.std()), 1e-6) if len(d) else 1e-6)

    settled: dict[str, int] = {}
    while free:
        total = sum(N * S for N, S in (stats[h] for h in free))
        pool = remaining - sum(settled.values())
        draft = {h: pool * (stats[h][0] * stats[h][1]) / total for h in free}
        clipped = False
        for h, v in list(draft.items()):
            N = stats[h][0]
            if v < MIN_PER_STRATUM:
                settled[h] = min(MIN_PER_STRATUM, N)
                del free[h]
                clipped = True
            elif v > N:
                settled[h] = N
                del free[h]
                clipped = True
        if not clipped:
            for h, v in draft.items():
                settled[h] = int(round(v))
            break
    alloc.update(settled)

    drift = budget - sum(alloc.values())
    if drift:
        candidates = [h for h in strata if h not in fixed]
        if not candidates:
            raise E390DesignError(
                "cannot allocate the residual budget: every observed "
                "stratum is fixed, so there is no non-fixed stratum left "
                f"to receive it (budget={budget}, fixed total="
                f"{sum(fixed.values())})")
        target = max(candidates, key=lambda h: len(strata[h]))
        alloc[target] = max(0, alloc[target] + drift)
    return alloc


def _draw(
    strata: dict[str, list[dict]],
    nominal: dict[str, int],
    homonym_group_by_id: dict[str, str],
    seed: int,
) -> tuple[set[str], dict[str, int]]:
    """One rng.default_rng(seed) instance, consumed sequentially across
    strata in `strata`'s own (sorted-key) iteration order: stratum
    iteration order is fixed because a single RNG is consumed sequentially
    across strata, so a different order would consume random draws
    differently and change membership. H_homonym draws whole groups by
    permutation until its nominal allocation is met or exceeded (never
    compensating another stratum for the overshoot); every other stratum
    -- including the take-all/capped hard strata -- draws via rng.choice
    without replacement, so that even a full-population "draw" still
    consumes RNG state the same way a partial one would.
    """
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    realised: dict[str, int] = {}

    for h, rows in strata.items():
        n = nominal[h]
        if h == "H_homonym":
            gids = sorted({homonym_group_by_id[r["entry_id"]] for r in rows})
            members: dict[str, list[str]] = defaultdict(list)
            for r in rows:
                members[homonym_group_by_id[r["entry_id"]]].append(r["entry_id"])
            order = rng.permutation(len(gids))
            got = 0
            for i in order:
                if got >= n:
                    break
                g = gids[i]
                selected.update(members[g])
                got += len(members[g])
            realised[h] = got
        else:
            ids = sorted(r["entry_id"] for r in rows)
            take = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
            selected.update(ids[i] for i in take)
            realised[h] = min(n, len(ids))
    return selected, realised


def design_e390(
    features: pa.Table,
    disagreement: pa.Table,
    *,
    budget: int = 390,
    seed: int = 20260821,
) -> E390DesignResult:
    """Reproduce the E390 stratified probability sample design.

    `features` is the canonical public phi table for the full sampling
    frame -- entry_id, homonym_index, para_idx, the hard-stratum predicate
    columns, and every phi_C/CLUSTER_FEATURES column. `disagreement` is a
    frozen archival design input -- entry_id + disagreement (an optional
    n_readers column is tolerated for provenance but never read) -- joined
    to `features` by entry_id; an entry_id absent from `disagreement`
    contributes no value to its stratum's Neyman variance estimate but is
    otherwise untouched. This function computes no disagreement itself and
    accepts no reader output.

    Raises E390DesignError on a malformed/inconsistent input, an
    impossible budget, or a violated internal invariant of the design
    procedure -- see module tests for the full gate list. Never returns a
    partial result alongside an exception.
    """
    if features.num_rows == 0:
        raise E390DesignError("cannot design E390 on an empty feature table")

    feature_names = set(features.schema.names)
    missing_cols = sorted(set(_REQUIRED_FEATURE_COLUMNS) - feature_names)
    if missing_cols:
        raise E390DesignError(
            f"feature table is missing required column(s): {missing_cols}")

    disagreement_names = set(disagreement.schema.names)
    if not {"entry_id", "disagreement"} <= disagreement_names:
        raise E390DesignError(
            "disagreement table must contain 'entry_id' and 'disagreement' "
            f"columns, got {sorted(disagreement_names)}")

    frame = features.to_pylist()
    ids_in_order = [f["entry_id"] for f in frame]
    feature_id_counts = Counter(ids_in_order)
    dup_feature_ids = sorted(i for i, n in feature_id_counts.items() if n > 1)
    if dup_feature_ids:
        raise E390DesignError(
            f"duplicate entry_id(s) in feature table: {dup_feature_ids}")
    feature_id_set = set(ids_in_order)

    dis_rows = disagreement.to_pylist()
    dis_ids = [r["entry_id"] for r in dis_rows]
    dis_id_counts = Counter(dis_ids)
    dup_dis_ids = sorted(i for i, n in dis_id_counts.items() if n > 1)
    if dup_dis_ids:
        raise E390DesignError(
            f"duplicate entry_id(s) in disagreement table: {dup_dis_ids}")

    unknown_dis_ids = sorted(set(dis_ids) - feature_id_set)
    if unknown_dis_ids:
        raise E390DesignError(
            "disagreement table has entry_id(s) not present in the feature "
            f"frame: {unknown_dis_ids}")

    dis_map: dict[str, float] = {}
    for r in dis_rows:
        v = r["disagreement"]
        if v is None or not math.isfinite(float(v)):
            raise E390DesignError(
                f"disagreement table entry {r['entry_id']!r} has a "
                f"non-finite or null disagreement value ({v!r}) -- an "
                "entry with no measurable disagreement must simply be "
                "absent from this table, not present with an invalid "
                "value")
        dis_map[r["entry_id"]] = float(v)

    for f in frame:
        f["disagreement"] = dis_map.get(f["entry_id"], float("nan"))

    # ── hard strata (first-match precedence; not altered) ───────────────
    for f in frame:
        f["stratum"] = _hard_stratum(f)

    hom_rows = [f for f in frame if f["stratum"] == "H_homonym"]
    homonym_group_by_id = _homonym_groups(hom_rows)

    # ── bulk population, canonical T, GMM ────────────────────────────────
    bulk = [f for f in frame if f["stratum"] is None]
    if not bulk:
        raise E390DesignError(
            "no bulk entries remain after hard-stratum assignment; cannot "
            "fit the structural transform/GMM on an empty bulk population")

    bulk_cols: dict[str, list] = {"entry_id": [f["entry_id"] for f in bulk]}
    for c in CLUSTER_FEATURES:
        bulk_cols[c] = [f[c] for f in bulk]
    bulk_table = pa.Table.from_pydict(bulk_cols)

    transform = fit_structural_transform(bulk_table, seed=seed, variance_threshold=0.90)
    projected = apply_structural_transform(transform, bulk_table)
    z_columns = [f"z_{j}" for j in range(transform.n_components)]
    Z = np.column_stack([projected.column(c).to_numpy() for c in z_columns])

    labels, k, gmm_diagnostics = _cluster_bulk(Z, seed)
    for f, lab in zip(bulk, labels):
        f["stratum"] = f"C{lab}"

    # ── assemble final strata ────────────────────────────────────────────
    for f in frame:
        if f["stratum"] is None:
            raise E390DesignError(
                f"entry {f['entry_id']!r} has no assigned stratum after "
                "hard-stratum and cluster assignment")

    strata: dict[str, list[dict]] = defaultdict(list)
    for f in frame:
        strata[f["stratum"]].append(f)
    strata = dict(sorted(strata.items()))

    fixed = {
        "H_multisense": len(strata.get("H_multisense", [])),
        "H_pointer": min(POINTER_CAP, len(strata.get("H_pointer", []))),
    }

    population = len(frame)
    if budget <= 0:
        raise E390DesignError(f"budget must be positive, got {budget}")
    if budget < sum(fixed.values()):
        raise E390DesignError(
            f"budget {budget} is smaller than the fixed-stratum "
            f"allocations alone ({sum(fixed.values())})")
    if budget > population:
        raise E390DesignError(
            f"budget {budget} exceeds the sampling frame's population "
            f"({population})")

    nominal = _neyman(strata, budget, fixed)

    for h, rows in strata.items():
        N_h = len(rows)
        n_h = nominal.get(h, 0)
        if n_h < 0 or n_h > N_h:
            raise E390DesignError(
                f"nominal allocation for stratum {h!r} is out of range: "
                f"n_h={n_h}, N_h={N_h}")

    selected, realised = _draw(strata, nominal, homonym_group_by_id, seed)

    unknown_selected = selected - feature_id_set
    if unknown_selected:
        raise E390DesignError(
            "internal error: selected entry_id(s) outside the feature "
            f"frame: {sorted(unknown_selected)}")

    realised_total = sum(realised.get(h, 0) for h in strata)
    if realised_total != len(selected):
        raise E390DesignError(
            "internal error: realised selected size is inconsistent with "
            f"the sum of per-stratum realised counts ({realised_total} "
            f"!= {len(selected)})")

    selected_entry_ids = tuple(eid for eid in ids_in_order if eid in selected)
    if len(set(selected_entry_ids)) != len(selected_entry_ids):
        raise E390DesignError(
            "internal error: duplicate entry_id in selected_entry_ids")

    design_rows = []
    for f in frame:
        st = f["stratum"]
        design_rows.append({
            "entry_id": f["entry_id"],
            "stratum": st,
            "homonym_group": homonym_group_by_id.get(f["entry_id"]),
            "selected": f["entry_id"] in selected,
            "N_h": len(strata[st]),
            "nominal_n_h": nominal.get(st, 0),
            "realised_n_h": realised.get(st, 0),
            "disagreement_available": f["entry_id"] in dis_map,
        })
    design_table = pa.Table.from_pylist(design_rows)

    report = E390DesignReport(
        seed=seed,
        budget=budget,
        frame_size=len(frame),
        bulk_size=len(bulk),
        selected_k=k,
        retained_pca_components=transform.n_components,
        explained_variance=float(sum(transform.explained_variance_ratio)),
        gmm_candidates=gmm_diagnostics,
        nominal_allocation=MappingProxyType(dict(nominal)),
        realised_allocation=MappingProxyType(dict(realised)),
        population_by_stratum=MappingProxyType({h: len(rows) for h, rows in strata.items()}),
        realised_selected_size=len(selected),
    )

    return E390DesignResult(
        selected_entry_ids=selected_entry_ids,
        transform=transform,
        report=report,
        design_table=design_table,
    )
