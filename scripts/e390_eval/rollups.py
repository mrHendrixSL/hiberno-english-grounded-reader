"""Phase 4: first-class rollups from the per-field-path metrics
`pipeline.score_one_condition` already computes -- every rollup here sums
raw true_positive/predicted_count/reference_count counts first and calls
`precision_recall_f1` exactly once (`scoring.rollup_from_field_metrics`);
none of them average per-field F1 values.
"""
from __future__ import annotations

from typing import Mapping

from . import contract
from .scoring import rollup_from_field_metrics

FIELD_GROUPS_FOR_ROLLUP: tuple[str, ...] = (
    "head_pron", "etymology", "senses", "cross_references", "usage_notes")


def compute_rollups(condition_result: Mapping[str, dict]) -> dict:
    """`condition_result` is one `pipeline.score_one_condition` result
    (has "extraction" and "resolver" per-field-path metrics dicts).
    Returns:

      targeted_extraction        -- A: fields the current reader
                                     architecture actually attempts
                                     (excludes sub_entries.* and
                                     usage_notes[].kind -- see
                                     contract.targeted_extraction_field_paths)
      full_schema_extraction     -- B: every adjudicated extraction field
                                     in the contract, gaps included
      field_groups               -- C: one P/R/F1 per of the five field
                                     groups, plus schema_coverage_gaps
      resolver                   -- D: resolver fields only, never mixed
                                     into any extraction rollup above
    """
    extraction = condition_result["extraction"]
    resolver = condition_result["resolver"]

    targeted = rollup_from_field_metrics(extraction, contract.targeted_extraction_field_paths())
    full_schema = rollup_from_field_metrics(extraction, contract.all_extraction_field_paths())

    field_groups = {
        group: rollup_from_field_metrics(extraction, contract.EXTRACTION_FIELD_PATHS[group])
        for group in FIELD_GROUPS_FOR_ROLLUP
    }
    field_groups["schema_coverage_gaps"] = rollup_from_field_metrics(
        extraction, contract.schema_coverage_gap_field_paths())

    resolver_rollup = rollup_from_field_metrics(resolver, tuple(resolver.keys()))

    return {
        "targeted_extraction": targeted,
        "full_schema_extraction": full_schema,
        "field_groups": field_groups,
        "resolver": resolver_rollup,
    }
