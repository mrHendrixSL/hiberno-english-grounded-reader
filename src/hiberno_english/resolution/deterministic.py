"""The deterministic post-generation resolution operator, rho.

rho fills two kinds of metadata that were never generated -- neither the
bare nor the grounded reader's raw output can carry a resolved identity,
by construction of the strict response contracts in
`hiberno_english.readers.llm.contracts` -- from source-side authority data
that is independent of any reader:

    Region.resolved_name / Region.source_type
        from a corpus-wide informant/bibliography authority lookup;

    CrossReference.resolution / .targets / .resolves
        from a corpus-wide, source-side headword index.

rho is a plain annotation pass over already-assembled `Entry` objects. It
never calls a model, never retrieves, and knows nothing about the B/R/D/G
reader conditions -- a future reader wrapper decides whether to apply it.
Its lookup space (the headword index, the informant/bibliography tables)
is built once from source-side data by `build_resolution_context` and
reused, unchanged, across every entry `resolve_entry` is applied to.

The headword index and the informant resolver reproduce the current
research resolvers' executable behaviour exactly, adapted to caller-
supplied `pyarrow.Table` input instead of reading Parquet internally:
matching is strict-first, then a loose word-division fallback, then a
small explicit verified pointer-alias table, with explicit non-pointer
exclusions; a shared headword or a shared informant code that the source
does not disambiguate is never guessed at -- it is returned in full
(`homonyms`) or left unresolved, never collapsed to one candidate.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import pyarrow as pa

from ..schema import Entry, Region

_REQUIRED_SEGMENTED_COLUMNS = ("entry_id", "entry_type", "raw", "homonym_index")
_REQUIRED_NON_LITERARY_COLUMNS = ("code", "county", "name", "source_type")
_REQUIRED_BIBLIOGRAPHY_COLUMNS = ("author", "title", "year")


class DeterministicResolutionError(RuntimeError):
    """Raised when a resolution context cannot be built from trustworthy
    input, on trustworthy input. No partial context is ever returned
    alongside this exception."""


# ── headword extraction/normalisation (source-side only) ───────────────────
# The printed apparatus that can terminate a headword: a pronunciation, a
# comma, or a part-of-speech label. Whichever comes first wins.
_POS = (r"n|v|adj|adv|int|prep|conj|exclam|phr|pron|pl|sb|vbl|attrib|"
        r"p\.part|n\.phr|v\.phr|adj\.phr|adv\.phr|num|art|interj")
_STOP = re.compile(rf"\s?/|,|\s(?:{_POS})\.|\ssee\s|\.\s", re.IGNORECASE)

# The source prints variants inline after the lemma: "gallus also gallous".
# The lemma is what precedes "also"/"or".
_ALSO = re.compile(r"\s+(?:also|or)\s+.*$", re.IGNORECASE)

# An inverted lemma keeps a comma INSIDE the headword ("Garden County,
# The"), indistinguishable from "lemma, see X" by punctuation alone, so
# both the truncated and the extended form are indexed as aliases.
_INVERTED_TAIL = re.compile(
    r",\s+(?:the|a|an|to\s+\w+|sit\s+\w+|be\s+\w+|go\s+\w+|of\s+\w+)\b.*",
    re.IGNORECASE)

# Trailing homonym numeral: "brock3", "tack4".
_HOM = re.compile(r"[0-9¹²³⁰-⁹]+$")
_SUPERSCRIPT_DIGITS = str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789")


def _strip_hom(s: str) -> str:
    return _HOM.sub("", s).strip()


def _norm(s: str) -> str:
    """Fold for matching: NFC, casefold, collapse whitespace, drop a
    trailing homonym numeral and the punctuation a small-caps run drags in."""
    s = unicodedata.normalize("NFC", s or "")
    s = s.strip().strip("’‘'\"“”().,;:!?-–—")
    s = _strip_hom(s)
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _extract_headword(raw: str) -> str:
    """The lemma as printed, before any pronunciation, comma, or POS label."""
    raw = (raw or "").strip()
    m = _STOP.search(raw)
    head = raw[: m.start()] if m else raw
    head = _ALSO.sub("", head)
    return _strip_hom(head.strip())


def _aliases(raw: str) -> set[str]:
    """Every form one source-side entry could legitimately be pointed at
    by: the plain headword, an inverted-lemma's truncated and extended
    forms, and a parenthesised element treated as optional."""
    raw = (raw or "").strip()
    out = {_extract_headword(raw)}

    m = re.search(rf"\s?/|\s(?:{_POS})\.|\ssee\s", raw, re.IGNORECASE)
    if m:
        extended = _ALSO.sub("", raw[: m.start()]).strip()
        if _INVERTED_TAIL.search(extended):
            out.add(_strip_hom(extended))
            out.add(_strip_hom(_INVERTED_TAIL.sub("", extended)))

    for a in list(out):
        if "(" in a:
            out.add(re.sub(r"\s*\([^)]*\)", "", a).strip())
            out.add(a.replace("(", "").replace(")", "").strip())

    return {a for a in out if a}


def _homonym_of(target: str) -> int | None:
    """The homonym numeral a cross-reference target carries, if any --
    recovered separately from `_norm`, which deliberately strips it."""
    t = unicodedata.normalize("NFC", target or "").strip().translate(_SUPERSCRIPT_DIGITS)
    m = re.search(r"([0-9])\s*$", t)
    return int(m.group(1)) if m else None


def _loose(s: str) -> str:
    """A weaker key: `_norm` with hyphens, apostrophes, and spaces removed
    -- used only as a fallback after the strict key misses, since
    collapsing word boundaries can manufacture matches that were never
    intended."""
    return re.sub(r"[\s\-’'‘]", "", _norm(s))


# ── small, explicit, manually verified rule sets ────────────────────────────
# Cross-reference targets the automatic index cannot reach, held as data
# with a reason category rather than folded into the matcher. Categories:
#   inflection   target is a plural or a different verb form of the lemma
#   spelling     target spells the lemma differently
#   truncated    the printed run is defective and cut the target short
#   subentry     target is printed inside another entry, not as a lemma
_POINTER_ALIASES: dict[str, tuple[str, str]] = {
    "spalpeens": ("ES0500", "inflection"),
    "eccers": ("EE0013", "inflection"),
    "leadar": ("EL0052", "inflection"),
    "yerrah": ("EY0009", "spelling"),
    "ná bac leish": ("EN0002", "spelling"),
    "aghrá": ("EA0035", "spelling"),
    "gealgháireach": ("EG0077", "spelling"),
    "dunduckety mudcoloured": ("ED0178", "spelling"),
    "national": ("EN0017", "truncated"),
    "trinity college": ("ET0154", "truncated"),
    "cute hoor": ("EH0077", "truncated"),
    "madra rua": ("EM0011", "subentry"),
}

# Targets that are not internal cross-references at all -- the source's own
# small-caps convention also marks bibliographic citation names.
_NOT_A_POINTER = frozenset({"edd", "hayden and hartog"})

# Localities the source prints that are finer than the county an authority
# row records, and lie inside it.
_WITHIN = {
    "forth and bargy": "wexford",
    "sliabh luachra": "kerry",
}


def _same_place(authority: str | None, printed: str | None) -> bool:
    a = (authority or "").strip().lower()
    p = (printed or "").strip().lower()
    if not a or not p:
        return False
    return p == a or _WITHIN.get(p) == a


@dataclass(frozen=True)
class DeterministicResolutionContext:
    """Precomputed, reusable state for `resolve_entry`: a source-side
    headword index (strict and loose), a homonym map, and informant/
    bibliography authority lookups. Built once by `build_resolution_context`
    and safe to reuse across every entry in a corpus -- `resolve_entry`
    never rebuilds any of it, rereads a file, or reparses an authority
    table.

    `strict_index` and `loose_index` map a normalised headword form to the
    tuple of entry_ids that print it, in source/corpus order -- a form
    shared by more than one entry stays multi-valued rather than being
    collapsed to one. `homonym_map` maps entry_id to its printed homonym
    index, for entries that have one. `informant_rows` maps an informant
    code to every authority row that carries it (never collapsed to one,
    since the same code can name several people). `bibliography_by_key`
    maps an author's full citation form and surname to its bibliography
    row.
    """

    strict_index: Mapping[str, tuple[str, ...]]
    loose_index: Mapping[str, tuple[str, ...]]
    homonym_map: Mapping[str, int]
    informant_rows: Mapping[str, tuple[Mapping[str, object], ...]]
    bibliography_by_key: Mapping[str, Mapping[str, object]]


def _validate_segmented_schema(segmented_entries: pa.Table) -> None:
    names = segmented_entries.schema.names
    if len(set(names)) != len(names):
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        raise DeterministicResolutionError(
            f"segmented_entries has duplicate column name(s): {dup}")
    missing = [c for c in _REQUIRED_SEGMENTED_COLUMNS if c not in names]
    if missing:
        raise DeterministicResolutionError(
            f"segmented_entries is missing required column(s): {missing}")


def _validate_segmented_rows(rows: list[dict]) -> None:
    ids: list[str] = []
    for i, r in enumerate(rows):
        eid = r["entry_id"]
        if not isinstance(eid, str) or not eid:
            raise DeterministicResolutionError(
                f"segmented_entries row {i}: entry_id must be a non-empty "
                f"string, got {eid!r}")
        ids.append(eid)

        etype = r["entry_type"]
        if not isinstance(etype, str) or not etype:
            raise DeterministicResolutionError(
                f"segmented_entries row {i} ({eid!r}): entry_type must be "
                f"a non-empty string, got {etype!r}")

        hom = r["homonym_index"]
        if hom is not None and (isinstance(hom, bool) or not isinstance(hom, int)):
            raise DeterministicResolutionError(
                f"segmented_entries row {i} ({eid!r}): homonym_index must "
                f"be None or an int, got {hom!r}")

        raw = r["raw"]
        if raw is not None and not isinstance(raw, str):
            raise DeterministicResolutionError(
                f"segmented_entries row {i} ({eid!r}): raw must be a "
                f"string or None, got {type(raw).__name__}")

    dup_ids = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dup_ids:
        raise DeterministicResolutionError(
            f"segmented_entries has duplicate entry_id(s): {dup_ids}")


def _validate_authority_schema(
        table: pa.Table, name: str, required: tuple[str, ...]) -> None:
    names = table.schema.names
    if len(set(names)) != len(names):
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        raise DeterministicResolutionError(
            f"authority table {name!r} has duplicate column name(s): {dup}")
    missing = [c for c in required if c not in names]
    if missing:
        raise DeterministicResolutionError(
            f"authority table {name!r} is missing required column(s): {missing}")


def _build_strict_index(lexical_rows: list[dict]) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for r in lexical_rows:
        for a in _aliases(r["raw"]):
            key = _norm(a)
            if key and r["entry_id"] not in idx.setdefault(key, []):
                idx[key].append(r["entry_id"])
    return idx


def _build_loose_index(strict_index: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, ids in strict_index.items():
        for e in ids:
            if e not in out.setdefault(_loose(k), []):
                out[_loose(k)].append(e)
    return out


def _build_bibliography_index(rows: list[dict]) -> dict[str, dict]:
    """Author full-citation-form/surname -> bibliography row, first-wins on
    a collision -- some source citations name a published work by author
    rather than by an informant code, so cross-reference resolution has to
    consult this table too."""
    out: dict[str, dict] = {}
    for row in rows:
        author = (row.get("author") or "").strip().strip("‘’'\"")
        if not author:
            continue
        surname = author.split(",")[0].strip()
        for key in {author, surname}:
            out.setdefault(key, row)
    return out


def build_resolution_context(
    segmented_entries: pa.Table,
    authority_tables: Mapping[str, pa.Table],
) -> DeterministicResolutionContext:
    """Build a reusable `DeterministicResolutionContext` from caller-
    supplied, already-public source-side data.

    `segmented_entries` (from `hiberno_english.segmentation`) must carry at
    least `entry_id`, `entry_type`, `raw`, `homonym_index`; letter headers
    (`entry_type == "letter_header"`) are excluded from the headword index.
    `authority_tables` (from `hiberno_english.apparatus`) must contain
    `non_literary_sources` and `bibliography`. Neither is read from a file
    or reparsed by this function's caller -- both are supplied as
    `pyarrow.Table` objects already in memory.

    Raises `DeterministicResolutionError` on a missing required column, a
    duplicate column name, a duplicate or empty `entry_id`, a malformed
    `homonym_index`/`entry_type`/`raw` value, or a missing required
    authority table -- never silently drops a malformed row.
    """
    _validate_segmented_schema(segmented_entries)
    seg_rows = segmented_entries.to_pylist()
    _validate_segmented_rows(seg_rows)

    if "non_literary_sources" not in authority_tables:
        raise DeterministicResolutionError(
            "authority_tables is missing required table 'non_literary_sources'")
    if "bibliography" not in authority_tables:
        raise DeterministicResolutionError(
            "authority_tables is missing required table 'bibliography'")

    non_literary = authority_tables["non_literary_sources"]
    bibliography = authority_tables["bibliography"]
    _validate_authority_schema(
        non_literary, "non_literary_sources", _REQUIRED_NON_LITERARY_COLUMNS)
    _validate_authority_schema(
        bibliography, "bibliography", _REQUIRED_BIBLIOGRAPHY_COLUMNS)

    lexical_rows = [r for r in seg_rows if r["entry_type"] != "letter_header"]

    strict_index = _build_strict_index(lexical_rows)
    loose_index = _build_loose_index(strict_index)
    homonym_map = {r["entry_id"]: r["homonym_index"] for r in lexical_rows
                  if r["homonym_index"] is not None}

    informant_rows: dict[str, list[dict]] = {}
    for row in non_literary.to_pylist():
        code = row.get("code")
        if not code:
            continue
        informant_rows.setdefault(code, []).append(row)

    bibliography_by_key = _build_bibliography_index(bibliography.to_pylist())

    return DeterministicResolutionContext(
        strict_index=MappingProxyType(
            {k: tuple(v) for k, v in strict_index.items()}),
        loose_index=MappingProxyType(
            {k: tuple(v) for k, v in loose_index.items()}),
        homonym_map=MappingProxyType(homonym_map),
        informant_rows=MappingProxyType(
            {k: tuple(v) for k, v in informant_rows.items()}),
        bibliography_by_key=MappingProxyType(dict(bibliography_by_key)),
    )


# ── cross-reference resolution ──────────────────────────────────────────────
def _resolve_target(target: str, context: DeterministicResolutionContext) -> tuple[str, ...]:
    """Entry ids a cross-reference target points at. Empty means it
    resolves to nothing; more than one means the form is shared by
    homonyms and the target does not say which -- returned in full,
    because picking one would invent a decision the source never made."""
    key = _norm(target)
    if key in _NOT_A_POINTER:
        return ()
    hits = list(context.strict_index.get(key, ()))
    n = _homonym_of(target)
    if n is not None and context.homonym_map and len(hits) > 1:
        narrowed = [e for e in hits if context.homonym_map.get(e) == n]
        if narrowed:
            return tuple(narrowed)
    if not hits:
        hits = list(context.loose_index.get(_loose(target), ()))
    if not hits and key in _POINTER_ALIASES:
        hits = [_POINTER_ALIASES[key][0]]
    return tuple(hits)


def _classify_resolution(target: str, hits: Sequence[str]) -> str:
    key = _norm(target)
    if key in _NOT_A_POINTER:
        return "not_a_pointer"
    alias = _POINTER_ALIASES.get(key)
    if alias is not None and alias[1] == "subentry":
        return "subentry"
    if not hits:
        return "dangling"
    return "unique" if len(hits) == 1 else "homonyms"


# ── informant/region resolution ─────────────────────────────────────────────
def _resolve_informant(
    code: str, county: str | None, context: DeterministicResolutionContext,
) -> Mapping[str, object] | None:
    """The authority row/synthesised bibliography entry for this citation,
    or None if it is not determined. A unique code still has to agree with
    the printed county -- uniqueness of the code says nothing about
    whether THIS citation is that source. A shared code resolves only when
    the printed county matches exactly one namesake."""
    rows = context.informant_rows.get(code) or ()
    if not rows:
        work = context.bibliography_by_key.get(code)
        if work is None:
            return None
        title = (work.get("title") or "").strip()
        year = work.get("year")
        yr = f" ({int(year)})" if year is not None else ""
        author = (work.get("author") or "").strip("‘’")
        return {"name": f"{author}, {title}{yr}".strip(", "),
               "source_type": "publication"}
    if len(rows) == 1:
        if not county or _same_place(rows[0].get("county"), county):
            return rows[0]
        return None
    if not county:
        return None
    hits = [r for r in rows if _same_place(r.get("county"), county)]
    return hits[0] if len(hits) == 1 else None


def _ground_region(region: Region, context: DeterministicResolutionContext) -> Region:
    if region.informant_code is None:
        return region
    row = _resolve_informant(region.informant_code.value, region.county, context)
    region.resolved_name = (row or {}).get("name")
    region.source_type = (row or {}).get("source_type")
    return region


def resolve_entry(entry: Entry, context: DeterministicResolutionContext) -> Entry:
    """Apply rho to one already-assembled `Entry`, returning a resolved
    deep copy -- `entry` itself is never mutated, so B/R may safely share
    the same raw generated `Entry` object that D/G apply this to.

    Fills only `Region.resolved_name`/`.source_type` (in every sense,
    gloss, and variant-form region) and `CrossReference.resolution`/
    `.targets`/`.resolves`. Every other field -- including
    `Region.informant_code`/`.county`/`.locus` and
    `CrossReference.target`/`.type` -- is copied through unchanged.
    Deterministic given `entry` and `context`; calling this again on an
    already-resolved `Entry` recomputes and overwrites the same
    resolution-only fields with the same values, never accumulating
    duplicate targets.
    """
    resolved = entry.model_copy(deep=True)

    for s in resolved.senses:
        s.regions = [_ground_region(r, context) for r in s.regions]
        for g in s.glosses:
            g.regions = [_ground_region(r, context) for r in g.regions]
    for vf in resolved.variant_forms:
        vf.regions = [_ground_region(r, context) for r in vf.regions]

    for x in resolved.cross_references:
        hits = _resolve_target(x.target.value, context)
        x.resolution = _classify_resolution(x.target.value, hits)
        x.targets = list(hits)
        x.resolves = bool(hits)

    return resolved
