"""Canonical source-side structural feature map (phi).

Builds one feature vector per lexical entry from the source material ONLY --
the segmented entry's raw paragraph text and its OOXML run styling. Nothing
here is derived from any reader's output, any adjudicated value, or any
downstream label: this representation is computed once, before any reader
runs, and is shared by every later stage that needs a structural view of an
entry (sample design, structural precedent retrieval) rather than being
owned by any one of them.

The features are deliberately close to the dictionary's own apparatus --
pronunciation, etymology, sense numbering, labels, informant codes,
cross-references, attestations -- because how many of these an entry's
source text and typography carry, and how densely, is a source-side proxy
for how much of the extraction schema that entry actually exercises. None
of this requires reading the definition for its MEANING; it is counting
structural and typographic evidence that is already there before any
reading is attempted.

Quote counts are PAIRS (count of marks // 2), not raw marks: a citation is
a pair of curly quotes, and counting marks would double every attestation
and silently reward entries whose quoting is unbalanced.

`entry_id`, `letter_section`, `homonym_index` and `para_idx` are carried
through as join/design metadata, not features: they identify or locate an
entry rather than describing how much structural apparatus it carries, and
must not be used as clustering or similarity input downstream.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pyarrow as pa

# ── source-side detectors ───────────────────────────────────────────────────
# Every pattern below keys off something the source does typographically or
# textually, never off what any reader produced.

IPA = re.compile(r"/[^/]{1,60}/")                    # pronunciation spans
ETYM_MARKER = re.compile(r"<")                       # an etymology derivation marker
SENSE_NUM = re.compile(r"(?:^|[;.]\s|\s)\d\.\s")     # numbered-sense markers ("1. ... 2. ...")
DQUOTE = re.compile(r"[“”]")               # literary citation convention
SQUOTE = re.compile(r"[‘’]")               # oral citation convention
REGION = re.compile(r"\(([A-Z]{2,4})(?:,\s*[A-Z][a-zA-Zà-ÿ' .-]+)?\)")
SEE_ONLY = re.compile(r"^[^,]{1,60},\s*see\s+[A-Z].{0,60}\.?$", re.I)
SEE_REF = re.compile(r"\bsee\s+[A-Z]", re.I)
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20[0-2]\d)\b")
# locus: a named-source page/line reference (e.g. "<Place>, 70:"), or a bare
# page.line pair such as "468.34" or "III, 72".
LOCUS = re.compile(r"\b\d+(?:\.\d+)?(?:[–-]\d+)?\s*:|\b\d+\.\d+\b")
IRISH_CHAR = re.compile(r"[áéíóúÁÉÍÓÚ]")
ALSO_VARIANT = re.compile(r"\balso\b", re.I)
CF_REF = re.compile(r"\bcf\.", re.I)
PROVERB = re.compile(r"\(proverb\)", re.I)
COLLOQ = re.compile(r"\(colloq\.\)|\(pejor\.\)|\(fig\.\)", re.I)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# a capitalised word that is NOT sentence-initial -- proper nouns, author
# names, work titles: the raw material of the attestation apparatus
CAPWORD = re.compile(r"(?<![.!?“‘]\s)(?<!^)\b[A-Z][a-zà-ÿ]{2,}")

# The count features, in a fixed order. Declared once so the Arrow schema
# and every consumer of this module agree on column identity and order.
COUNT_FEATURES: tuple[str, ...] = (
    "n_chars", "n_words",
    "n_ipa", "n_etym_marker", "n_sense_num",
    "n_dquote_pair", "n_squote_pair", "n_region_code",
    "n_xref_run", "n_italic_run", "n_smallcaps_run", "n_allcaps_run",
    "n_run", "n_char_style",
    "n_paren", "n_semicolon", "n_comma", "n_period", "n_digit",
    "n_year", "n_locus", "n_capword", "n_irish_char",
    "n_also", "n_cf", "n_see", "n_proverb", "n_register_paren",
)
BOOL_FEATURES: tuple[str, ...] = ("is_cross_ref", "has_homonym", "is_pointer")
FEATURE_COLUMNS: tuple[str, ...] = COUNT_FEATURES + BOOL_FEATURES

_METADATA_COLUMNS: tuple[str, ...] = ("letter_section", "homonym_index", "para_idx")


class StructuralFeatureError(RuntimeError):
    """Raised when the supplied entries/runs cannot produce a trustworthy
    structural feature table -- duplicate entry identifiers, duplicate
    para_idx among lexical entries, an entry with no para_idx to join
    against, an entry whose para_idx has no matching main-document run
    evidence, or an internal inconsistency between the lexical frame and
    the rows produced from it. No partial or silently-incomplete feature
    table is ever returned alongside this exception.
    """


def _build_run_index(runs: pa.Table) -> dict[int, list[dict]]:
    """Group runs by paragraph index, main document body only.

    Headers, footers, footnotes and endnotes have their own independent
    para_idx numbering. Consuming only "word/document.xml" here is what
    guarantees a header/footer paragraph that happens to share a
    dictionary entry's para_idx can never contribute run-derived features
    to that entry.
    """
    idx: dict[int, list[dict]] = defaultdict(list)
    for r in runs.to_pylist():
        if r["part"] != "word/document.xml":
            continue
        idx[r["para_idx"]].append(r)
    return idx


def _feature_row(e: dict, runs: list[dict]) -> dict:
    """One entry -> one flat row of every declared feature, plus metadata."""
    raw = e["raw"]
    f: dict = {"entry_id": e["entry_id"]}

    f["n_chars"] = len(raw)
    f["n_words"] = len(WORD.findall(raw))
    f["n_ipa"] = len(IPA.findall(raw))
    f["n_etym_marker"] = len(ETYM_MARKER.findall(raw))
    f["n_sense_num"] = len(SENSE_NUM.findall(raw))
    f["n_dquote_pair"] = len(DQUOTE.findall(raw)) // 2
    f["n_squote_pair"] = len(SQUOTE.findall(raw)) // 2
    f["n_region_code"] = len(REGION.findall(raw))

    # Run styling is structural evidence available before any reader runs:
    # how much typographic apparatus (etymon/POS italics, cross-reference
    # styling) an entry carries is a source-side proxy for how much of the
    # extraction schema that entry actually exercises.
    f["n_xref_run"] = sum(1 for r in runs if r["char_style"] == "1Text")
    f["n_italic_run"] = sum(1 for r in runs if r["italic"])
    f["n_smallcaps_run"] = sum(1 for r in runs if r["small_caps"])
    f["n_allcaps_run"] = sum(1 for r in runs if r["all_caps"])
    f["n_run"] = len(runs)
    f["n_char_style"] = len({r["char_style"] for r in runs if r["char_style"]})

    f["n_paren"] = raw.count("(")
    f["n_semicolon"] = raw.count(";")
    f["n_comma"] = raw.count(",")
    f["n_period"] = raw.count(".")
    f["n_digit"] = sum(c.isdigit() for c in raw)

    f["n_year"] = len(YEAR.findall(raw))
    f["n_locus"] = len(LOCUS.findall(raw))
    f["n_capword"] = len(CAPWORD.findall(raw))
    f["n_irish_char"] = len(IRISH_CHAR.findall(raw))

    f["n_also"] = len(ALSO_VARIANT.findall(raw))
    f["n_cf"] = len(CF_REF.findall(raw))
    f["n_see"] = len(SEE_REF.findall(raw))
    f["n_proverb"] = len(PROVERB.findall(raw))
    f["n_register_paren"] = len(COLLOQ.findall(raw))

    f["is_cross_ref"] = e["entry_type"] == "cross_ref"
    f["has_homonym"] = e["homonym_index"] is not None
    f["is_pointer"] = bool(SEE_ONLY.match(raw.strip()))

    # Carried through as design/join metadata, not clustering input: see
    # the module docstring.
    f["letter_section"] = e["letter_section"]
    f["homonym_index"] = e["homonym_index"]
    f["para_idx"] = e["para_idx"]
    return f


def build_structural_features(entries: pa.Table, runs: pa.Table) -> pa.Table:
    """Build the canonical structural feature table phi, one row per
    lexical entry (every `entries` row with `entry_type != "letter_header"`).

    `entries` and `runs` must be source-side only: segmentation's entry
    table and ingestion's run table. This function does not accept, and
    must never be extended to accept, reader output, adjudicated
    references, or any other post-reading label.

    Raises `StructuralFeatureError` if the lexical frame contains a
    duplicate entry_id, an entry with no para_idx to join run data
    against, two lexical entries sharing one para_idx (segmentation's
    contract is one lexical entry per paragraph), an entry whose para_idx
    has no matching word/document.xml run evidence at all, or if the
    constructed rows are internally inconsistent with that frame. No
    partial result is ever returned alongside this exception -- in
    particular, a paragraph missing from the run index is never silently
    treated as a legitimate all-zero run-style feature vector.
    """
    entry_rows = entries.to_pylist()
    frame = [e for e in entry_rows if e["entry_type"] != "letter_header"]

    ids = [e["entry_id"] for e in frame]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    if dup_ids:
        raise StructuralFeatureError(
            f"duplicate lexical entry_id(s), cannot build a trustworthy "
            f"feature table: {dup_ids}")

    unjoinable = [e["entry_id"] for e in frame if e.get("para_idx") is None]
    if unjoinable:
        raise StructuralFeatureError(
            f"lexical entries with no para_idx to join against run data: {unjoinable}")

    para_to_ids: dict[int, list[str]] = defaultdict(list)
    for e in frame:
        para_to_ids[e["para_idx"]].append(e["entry_id"])
    dup_para = {p: eids for p, eids in para_to_ids.items() if len(eids) > 1}
    if dup_para:
        raise StructuralFeatureError(
            "duplicate para_idx among lexical entries, violating the "
            "one-lexical-entry-per-paragraph segmentation contract: "
            f"{dict(sorted(dup_para.items()))}")

    run_index = _build_run_index(runs)
    missing_runs = {e["entry_id"]: e["para_idx"] for e in frame
                    if e["para_idx"] not in run_index}
    if missing_runs:
        raise StructuralFeatureError(
            "lexical entries with no matching word/document.xml run(s) -- "
            "missing structural run evidence must not silently become a "
            f"zero-valued feature vector: {missing_runs}")

    feature_rows = [_feature_row(e, run_index[e["para_idx"]]) for e in frame]

    if len(feature_rows) != len(frame):
        raise StructuralFeatureError(
            "internal error: feature-row count does not match the lexical frame")

    columns = ("entry_id", *FEATURE_COLUMNS, *_METADATA_COLUMNS)
    for row in feature_rows:
        for name in FEATURE_COLUMNS:
            if row.get(name) is None:
                raise StructuralFeatureError(
                    f"entry {row.get('entry_id')!r} is missing feature {name!r}")

    ordered_rows = [{c: r[c] for c in columns} for r in feature_rows]
    if [r["entry_id"] for r in ordered_rows] != ids:
        raise StructuralFeatureError(
            "internal error: output entry_id order does not match the "
            "lexical-entry input order")

    return pa.Table.from_pylist(ordered_rows)
