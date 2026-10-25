"""H390: the held-out sample drawn from D \\ E390, reusing the FROZEN
E390 stratum/cluster assignments and homonym groups from
`data/05_e390/design_table.parquet` -- this module never fits or refits
the structural transform T, never fits a Gaussian mixture, never
recomputes reader disagreement, and never invents a new stratum. It
consumes ONLY `e390_design_table`'s own `entry_id`/`stratum`/
`homonym_group`/`selected` columns (canonical corpus order, exactly as
written by `hiberno_english.sampling.e390.design_e390`) plus a caller-
supplied nominal allocation and seed.

The seeded, whole-homonym-group draw mechanics are the EXACT SAME
function E390's own design used -- `hiberno_english.sampling.e390._draw`
-- imported and called unchanged, never reimplemented, so H390's draw
behaves identically to E390's own for the same kind of input. H390 does
NOT run Neyman allocation itself: `nominal_allocation` is an explicit,
externally-decided redistribution (this module performs no arithmetic
of its own to derive it), because E390 exhausted one stratum
(`H_multisense`, a 100%-take-all protected stratum -- see `e390.py`'s
own `_neyman`) entirely, leaving zero of it in the remaining population.

There is no post-draw override mechanism: `_draw`'s own realised size
either equals `budget` exactly or `draw_h390` raises. A whole-homonym
group is never split -- `nominal_allocation` must be chosen (by the
caller, ahead of time) so that `_draw`'s own realised totals reconcile
to `budget` without needing one.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pyarrow as pa

from .e390 import _draw

_REQUIRED_COLUMNS = ("entry_id", "stratum", "homonym_group", "selected")


class H390DesignError(RuntimeError):
    """Raised when H390 cannot be drawn on trustworthy input, when the
    caller-supplied nominal allocation is inconsistent with the actual
    remaining population, or when an internal invariant of the draw
    procedure is violated. No partial `H390DesignResult` is ever
    returned alongside this exception."""


@dataclass(frozen=True)
class H390DesignReport:
    seed: int
    budget: int
    frame_size: int
    nominal_allocation: MappingProxyType
    realised_allocation: MappingProxyType
    population_by_stratum: MappingProxyType
    excluded_stratum: str | None
    excluded_stratum_reason: str | None
    realised_selected_size: int


@dataclass(frozen=True)
class H390DesignResult:
    """`selected_entry_ids` is ordered exactly as `e390_design_table` was
    ordered (canonical corpus order), never as an unordered set.
    `design_table` carries one row per REMAINING (non-E390) entry."""

    selected_entry_ids: tuple[str, ...]
    design_table: pa.Table
    report: H390DesignReport

    def ordered_membership_sha256(self) -> str:
        return hashlib.sha256("\n".join(self.selected_entry_ids).encode("utf-8")).hexdigest()


def draw_h390(
    e390_design_table: pa.Table,
    *,
    nominal_allocation: Mapping[str, int],
    seed: int,
    excluded_stratum: str | None = None,
    excluded_stratum_reason: str | None = None,
    budget: int = 390,
) -> H390DesignResult:
    """Draw H390 from the entries `e390_design_table` marks `selected=
    False` (D \\ E390), grouped by their ALREADY-FROZEN `stratum` column
    -- never a freshly computed one.

    Raises `H390DesignError` if:
      - `e390_design_table` is missing a required column, has a duplicate
        `entry_id`, or has no remaining (non-selected) rows at all;
      - `excluded_stratum` is given but still has remaining population
        (it is not actually exhausted -- refuses to silently drop a
        non-empty stratum from H);
      - the remaining population contains a stratum with no corresponding
        key in `nominal_allocation` (refuses to invent an allocation);
      - any stratum's nominal allocation is negative or exceeds its own
        remaining population (the caller must supply an allocation the
        remaining frame can actually support -- this function never
        silently caps it down);
      - the realised size `_draw` actually produces does not equal
        `budget` exactly -- there is no override mechanism to reconcile
        a mismatch; the caller must choose a `nominal_allocation` (and,
        for a whole-group stratum, one whose group sizes actually permit
        landing on that nominal total) that reconciles cleanly.
    """
    missing_cols = set(_REQUIRED_COLUMNS) - set(e390_design_table.schema.names)
    if missing_cols:
        raise H390DesignError(f"e390_design_table is missing required column(s): {sorted(missing_cols)}")

    rows = e390_design_table.to_pylist()
    canonical_order = [r["entry_id"] for r in rows]
    if len(set(canonical_order)) != len(canonical_order):
        raise H390DesignError("duplicate entry_id in e390_design_table")

    remaining = [r for r in rows if not r["selected"]]
    if not remaining:
        raise H390DesignError("e390_design_table has no remaining (non-selected) entries -- nothing to draw H from")

    strata: dict[str, list[dict]] = defaultdict(list)
    for r in remaining:
        strata[r["stratum"]].append(r)
    strata = dict(sorted(strata.items()))

    if excluded_stratum is not None and excluded_stratum in strata:
        raise H390DesignError(
            f"excluded_stratum {excluded_stratum!r} still has "
            f"{len(strata[excluded_stratum])} remaining entries -- it is "
            "not actually exhausted; refusing to silently exclude a "
            "non-empty stratum from H")

    unexpected = set(strata) - set(nominal_allocation)
    if unexpected:
        raise H390DesignError(
            f"remaining population has stratum/strata {sorted(unexpected)} "
            "with no nominal H allocation supplied -- refusing to silently "
            "give them zero or invent an allocation")

    for h, rows_h in strata.items():
        n = nominal_allocation[h]
        if n < 0:
            raise H390DesignError(f"stratum {h!r}: nominal allocation must be >= 0, got {n}")
        if n > len(rows_h):
            raise H390DesignError(
                f"stratum {h!r}: nominal allocation {n} exceeds remaining "
                f"population {len(rows_h)}")

    homonym_group_by_id = {
        r["entry_id"]: r["homonym_group"] for r in remaining if r["stratum"] == "H_homonym"}

    selected, realised = _draw(strata, dict(nominal_allocation), homonym_group_by_id, seed)
    realised = dict(realised)

    if len(selected) != budget:
        raise H390DesignError(
            f"realised H size is {len(selected)}, expected exactly "
            f"budget={budget} -- refusing to freeze a size mismatch; there is "
            f"no override mechanism, so nominal_allocation={dict(nominal_allocation)} "
            f"must be revised (realised allocation was {realised})")

    selected_entry_ids = tuple(eid for eid in canonical_order if eid in selected)
    if len(selected_entry_ids) != len(selected):
        raise H390DesignError("internal error: canonical-order filtering lost or duplicated a selection")

    design_rows = [
        {
            "entry_id": r["entry_id"],
            "stratum": r["stratum"],
            "homonym_group": r["homonym_group"],
            "selected": r["entry_id"] in selected,
            "N_h": len(strata[r["stratum"]]),
            "nominal_n_h": nominal_allocation.get(r["stratum"], 0),
            "realised_n_h": realised.get(r["stratum"], 0),
        }
        for r in remaining
    ]
    design_table = pa.Table.from_pylist(design_rows)

    report = H390DesignReport(
        seed=seed,
        budget=budget,
        frame_size=len(remaining),
        nominal_allocation=MappingProxyType(dict(nominal_allocation)),
        realised_allocation=MappingProxyType(realised),
        population_by_stratum=MappingProxyType({h: len(rows_h) for h, rows_h in strata.items()}),
        excluded_stratum=excluded_stratum,
        excluded_stratum_reason=excluded_stratum_reason,
        realised_selected_size=len(selected),
    )

    return H390DesignResult(selected_entry_ids=selected_entry_ids, design_table=design_table, report=report)
