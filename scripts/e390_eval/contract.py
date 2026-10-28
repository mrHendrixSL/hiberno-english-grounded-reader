"""The E390 evaluation contract: an explicit, field-by-field classification
of the current public `hiberno_english.schema.Entry` model, and the
deterministic path-extraction primitive every metric in this package is
built from.

This classification is derived from repository evidence, not assumed:

  * `hiberno_english.readers.llm.assembly.FIELD_OWNERSHIP` and
    `assemble_entry` are the authority on which of the five field groups
    generates which Entry field, and confirm that `CrossReference.resolves/
    .resolution/.targets`, `Region.resolved_name/.source_type`, and
    `UsageNote.kind` are always left at their schema defaults by every LLM
    reader -- never generated.
  * `hiberno_english.readers.rule_based.extractor.extract_rule_based_entry`'s
    own docstring confirms RB_raw never populates `usage_notes` or
    `sub_entries` at all ("matching current executable behaviour, which
    extracts neither").
  * `hiberno_english.resolution.deterministic.resolve_entry` (and the
    identical `_masked_dump` convention already used at both the B/R/D/G
    factorial-composition boundary in `readers.llm.factorial` and the
    RB/RB_raw boundary in `readers.rule_based.reader`) is the authority on
    exactly which fields rho is permitted to touch: `Region.resolved_name`/
    `.source_type` and `CrossReference.resolution`/`.targets`/`.resolves`.
    Nothing else. This module reuses that exact field list rather than
    re-deriving it.
  * `hiberno_english.schema.models.Sourced`'s own docstring ("per-value
    provenance envelope") is the authority for a schema-wide rule applied
    uniformly here: for every `Sourced[T]` field anywhere in the schema,
    only `.value` is evaluable content -- `source_span`, `tier`,
    `decided_by`, `authority_ref`, and `status` are always
    metadata/provenance, excluded from correctness scoring.

Two methodological decisions below are NOT discovered facts and are
flagged as such rather than silently assumed -- see
`FLATTENED_MULTISET_RATIONALE` and `NO_CONTENT_NORMALIZATION_RATIONALE`.
"""
from __future__ import annotations

from typing import Mapping, Sequence

# ── the two explicit methodological decisions this module makes ────────────
#
# No existing convention in this repository resolves either of the
# following; both are new decisions, made explicitly and documented here
# rather than silently, per this build's own instructions.
FLATTENED_MULTISET_RATIONALE = (
    "No field in the current schema has a stable, corpus-independent "
    "identity key for a repeated substructure across independent readings "
    "(Sense.number and VariantForm/Pronunciation/Etymology.sense_number are "
    "documented as usually unset -- 'the large majority of entries print a "
    "single unnumbered sense'). Aligning predicted_item[i] to "
    "reference_item[j] by list position, or by a best-match heuristic, "
    "would therefore assert an identity the source material does not "
    "assert, or would require exactly the fuzzy/similarity matching this "
    "build is explicitly forbidden from introducing. This evaluator instead "
    "flattens every repeated structure to the unordered MULTISET of its "
    "leaf values at each field path, per entry, and scores that multiset "
    "against the reference's multiset at the same path. This is fully "
    "deterministic and requires no item-to-item alignment of any kind, but "
    "it is a real methodological choice: it cannot distinguish 'right "
    "content in the wrong sense' from 'right content in the right sense'. "
    "Flag for confirmation: if a future adjudication convention supplies a "
    "stable per-item key, this is the one file to revisit."
)
NO_CONTENT_NORMALIZATION_RATIONALE = (
    "The only string-normalisation convention in this codebase "
    "(hiberno_english.resolution.deterministic._norm) is scoped narrowly to "
    "headword-pointer MATCHING (NFC + casefold + punctuation-strip, "
    "deliberately lossy so a cross-reference target finds its referent) "
    "and is not a general content-equality convention -- reusing it here "
    "would silently fold case/punctuation distinctions this build is "
    "explicitly told not to strip. No other normalisation of extracted "
    "content exists anywhere in ingest/segmentation/extraction. This "
    "evaluator therefore applies NO normalisation to string content: "
    "comparison is exact string equality, including case, punctuation, and "
    "whitespace. Non-string scalars (bool, int, enum literals) use plain "
    "`==`. This is the conservative, zero-invention default, not a "
    "discovered convention."
)


class ContractError(RuntimeError):
    """Raised when a field path cannot be extracted, or a caller asks for
    a field path this contract does not know about."""


# ── generic, deterministic path extraction ──────────────────────────────────
def extract_path(entry: Mapping[str, object] | None, path: str) -> list[object]:
    """Extract every leaf value reachable via `path` from one Entry-shaped
    dict (as `Entry.model_dump(mode="json")` produces), as a flat list --
    the multiset this whole package scores against.

    `path` is a dotted sequence of keys; a `[]` suffix on a segment means
    "this key holds a list -- iterate it", e.g.
    `"senses[].attestations[].quote"` walks `entry["senses"]` (a list),
    each item's `["attestations"]` (a list), and collects every item's
    `["quote"]`. A `None` value anywhere along the walk simply contributes
    nothing (never an error) -- this is how an absent optional field or an
    empty list is represented, not a malformed input.

    Deterministic and side-effect free: never mutates `entry`, never
    sorts, never deduplicates -- duplicate leaf values are preserved as
    duplicates in the returned multiset.
    """
    if not path:
        raise ContractError("extract_path: path must be non-empty")
    values: list[object] = [entry]
    for token in path.split("."):
        is_list = token.endswith("[]")
        key = token[:-2] if is_list else token
        if not key:
            raise ContractError(f"extract_path: malformed path segment in {path!r}")
        next_values: list[object] = []
        for v in values:
            if not isinstance(v, Mapping):
                continue
            item = v.get(key)
            if item is None:
                continue
            if is_list:
                if not isinstance(item, list):
                    raise ContractError(
                        f"extract_path: path {path!r} expects a list at "
                        f"{key!r} but found {type(item).__name__}")
                next_values.extend(x for x in item if x is not None)
            else:
                next_values.append(item)
        values = next_values
    return values


# ── the field classification itself ─────────────────────────────────────────
#
# Category labels, exactly as required by this build's task text:
EVALUABLE_ATOMIC = "evaluable atomic/scalar"
EVALUABLE_COLLECTION = "evaluable collection"
EVALUABLE_ORDERED = "evaluable ordered structure"
EVALUABLE_NESTED = "evaluable nested structure"
METADATA_EXCLUDED = "metadata/provenance excluded from correctness"
RESOLVER_ANNOTATION = "resolver annotation scored separately"
NOT_APPLICABLE = "not applicable / intentionally excluded"

# Extraction-content field paths, grouped by the field group that owns
# them (`readers.llm.assembly.FIELD_OWNERSHIP`) -- every path here is
# EVALUABLE_NESTED except where noted, and every one already excludes the
# Sourced envelope (only `.value` is ever included) per the schema-wide
# rule above. RB owns the same schema fields via
# `readers.rule_based.extractor`/`reader.finalize_rule_based_entry`, so the
# identical path set applies to RB without modification.
EXTRACTION_FIELD_PATHS: Mapping[str, tuple[str, ...]] = {
    "head_pron": (
        "headword.value",
        "pos[].term.value",
        "pos[].kind",
        "pronunciations[].ipa.value",
        "pronunciations[].regional_qualifier",
        "pronunciations[].sense_number",
        "variant_forms[].form.value",
        "variant_forms[].qualifier",
        "variant_forms[].sense_number",
        "variant_forms[].regions[].informant_code.value",
        "variant_forms[].regions[].county",
        "variant_forms[].regions[].locus",
        "variant_forms[].pronunciations[].ipa.value",
        "variant_forms[].pronunciations[].regional_qualifier",
        "variant_forms[].pronunciations[].sense_number",
    ),
    "etymology": (
        "etymology[].language_code.value",
        "etymology[].etymon_form",
        "etymology[].gloss",
        "etymology[].attributed_to",
        "etymology[].chain_step",
        "etymology[].relation",
        "etymology[].etymon_pos",
        "etymology[].sense_number",
    ),
    "senses": (
        "senses[].number",
        "senses[].definition.value",
        "senses[].definition_attribution",
        "senses[].labels[].term.value",
        "senses[].labels[].kind",
        "senses[].regions[].informant_code.value",
        "senses[].regions[].county",
        "senses[].regions[].locus",
        "senses[].attestations[].quote",
        "senses[].attestations[].citation_class",
        "senses[].attestations[].source",
        "senses[].attestations[].locus",
        "senses[].attestations[].genre",
        "senses[].glosses[].text.value",
        "senses[].glosses[].labels[].term.value",
        "senses[].glosses[].labels[].kind",
        "senses[].glosses[].regions[].informant_code.value",
        "senses[].glosses[].regions[].county",
        "senses[].glosses[].regions[].locus",
    ),
    "cross_references": (
        # target/type only -- resolves/resolution/targets are
        # RESOLVER_FIELD_PATHS below, never mixed in here.
        "cross_references[].target.value",
        "cross_references[].type",
    ),
    "usage_notes": (
        "usage_notes[].text.value",
        "usage_notes[].attributed_to",
        # `kind` is real adjudicated content (set on 38/390 adjudicated
        # notes) but is DELIBERATELY never generated by any LLM reader
        # (assembly.py: "kind left at schema default (None): deliberately
        # never generated") and never generated by RB (extractor.py never
        # populates usage_notes at all). Included for transparency; every
        # current reader will show 0 recall here by construction, not as a
        # per-model defect.
        "usage_notes[].kind",
    ),
    # `sub_entries` is owned by NONE of the five field groups (assembly.py:
    # "`sub_entries` is likewise left at its empty default: none of the
    # five groups own it") and is never populated by RB either
    # (extractor.py: "`Entry.usage_notes` and `.sub_entries` stay at their
    # empty defaults, matching current executable behaviour, which
    # extracts neither"). 14/390 adjudicated entries DO have real
    # sub_entries content (confirmed by direct inspection of
    # frozen/eval390_reference.jsonl). Included for gap-reporting only:
    # RB and every LLM condition (B/R/D/G) will show 0 recall here by
    # construction -- a pre-existing, known reader-capability gap, not a
    # per-model or per-run defect this evaluation introduces or hides.
    "sub_entries": (
        "sub_entries[].form.value",
        "sub_entries[].pos[].term.value",
        "sub_entries[].pos[].kind",
        "sub_entries[].pronunciations[].ipa.value",
        "sub_entries[].senses[].definition.value",
        "sub_entries[].etymology[].etymon_form",
        "sub_entries[].usage_notes[].text.value",
    ),
}

# Resolver-produced annotations -- filled only by
# `hiberno_english.resolution.deterministic.resolve_entry`, identical to
# the exact field list `_masked_dump` already masks at both the B/R/D/G
# and RB/RB_raw composition boundaries (factorial.py, rule_based/reader.py).
# Never mixed into EXTRACTION_FIELD_PATHS' precision/recall.
RESOLVER_FIELD_PATHS: tuple[str, ...] = (
    "senses[].regions[].resolved_name",
    "senses[].regions[].source_type",
    "senses[].glosses[].regions[].resolved_name",
    "senses[].glosses[].regions[].source_type",
    "variant_forms[].regions[].resolved_name",
    "variant_forms[].regions[].source_type",
    "cross_references[].resolves",
    "cross_references[].resolution",
    "cross_references[].targets[]",
)

# Metadata/provenance: never scored under any condition.
#   - entry_id, entry_type, homonym_index, provenance: workflow/source
#     metadata supplied to every reader, never reader-generated
#     (assembly.py's own docstring for entry_id/entry_type/homonym_index/
#     provenance: "never generated by the language model"; the same four
#     are supplied identically to the RB reader and to the adjudicated
#     reference's own provenance).
#   - Every Sourced[T] envelope field except `.value` (source_span, tier,
#     decided_by, authority_ref, status), on EVERY Sourced field in the
#     schema -- a single schema-wide rule, not enumerated per field.
EXCLUDED_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "entry_id", "entry_type", "homonym_index", "provenance",
)
EXCLUDED_SOURCED_ENVELOPE_KEYS: tuple[str, ...] = (
    "source_span", "tier", "decided_by", "authority_ref", "status",
)

# Structural (count-based) diagnostics -- top-level repeated-field counts,
# reported per entry per condition, independent of the field-level P/R/F1
# metrics above. `len()` of the corresponding list; never a P/R/F1 value.
STRUCTURAL_COUNT_FIELDS: tuple[str, ...] = (
    "pos", "variant_forms", "pronunciations", "etymology",
    "senses", "cross_references", "usage_notes", "sub_entries",
)
# A derived structural count one level down, reported alongside the above.
STRUCTURAL_NESTED_COUNT_PATHS: Mapping[str, str] = {
    "attestations_total": "senses[].attestations[]",
    "glosses_total": "senses[].glosses[]",
}


def all_extraction_field_paths() -> tuple[str, ...]:
    """Every EVALUABLE_NESTED extraction-content path across all five
    field groups plus sub_entries, in a fixed, deterministic order."""
    paths: list[str] = []
    for group in ("head_pron", "etymology", "senses", "cross_references",
                  "usage_notes", "sub_entries"):
        paths.extend(EXTRACTION_FIELD_PATHS[group])
    return tuple(paths)


# ── targeted vs. full-schema-coverage rollup membership ─────────────────────
#
# "Targeted" = fields the current reader ARCHITECTURE actually attempts to
# generate, per direct code evidence (not per specific model's behaviour):
#   - `sub_entries` is owned by none of the five LLM field groups
#     (assembly.py) and is never populated by RB either (extractor.py) --
#     no part of the current architecture ever attempts it.
#   - `usage_notes[].kind` specifically (not the rest of usage_notes) is
#     deliberately never assigned by the LLM's own usage_notes field group
#     (assembly.py: "kind left at schema default (None): deliberately
#     never generated") and RB never populates usage_notes at all.
# These are the same two candidates this build's own task text names,
# confirmed here against current code rather than assumed.
_TARGETED_EXCLUDED_GROUPS: tuple[str, ...] = ("sub_entries",)
_TARGETED_EXCLUDED_PATHS: tuple[str, ...] = ("usage_notes[].kind",)


def targeted_extraction_field_paths() -> tuple[str, ...]:
    """`all_extraction_field_paths()` minus every path the current reader
    architecture (RB extractor + the five LLM field groups) never
    attempts to generate at all -- see `_TARGETED_EXCLUDED_GROUPS`/
    `_TARGETED_EXCLUDED_PATHS` for exactly which, and why, with a direct
    code citation for each. Applied uniformly to RB and every LLM
    condition so the rollup stays directly comparable across readers;
    where a reader additionally fails to attempt something still IN this
    targeted set (e.g. RB never populates usage_notes at all, even though
    `usage_notes[].text.value` is architecturally targeted because the
    LLM's own field group does attempt it), that shows up honestly as 0
    recall for that reader, rather than being hidden by narrowing the
    contract per reader.
    """
    return tuple(
        p for p in all_extraction_field_paths()
        if field_group_for_path(p) not in _TARGETED_EXCLUDED_GROUPS
        and p not in _TARGETED_EXCLUDED_PATHS
    )


def schema_coverage_gap_field_paths() -> tuple[str, ...]:
    """The complement of `targeted_extraction_field_paths()` within
    `all_extraction_field_paths()` -- exactly the fields no part of the
    current reader architecture attempts, kept visible as their own
    rollup rather than silently dropped."""
    targeted = set(targeted_extraction_field_paths())
    return tuple(p for p in all_extraction_field_paths() if p not in targeted)


def field_group_for_path(path: str) -> str:
    """Which of the six groups (five field groups + sub_entries) owns a
    given extraction-content field path. Raises `ContractError` for a
    resolver path or an unknown path -- callers must not conflate the
    two."""
    for group, paths in EXTRACTION_FIELD_PATHS.items():
        if path in paths:
            return group
    raise ContractError(f"{path!r} is not a known extraction field path")
