"""Deterministic rule-based lexical extraction: raw source paragraph text
plus its run styling, mapped onto an UNRESOLVED current-schema Entry.

This module produces RB_raw only. It never performs cross-reference or
informant identity resolution -- that is `hiberno_english.resolution.
resolve_entry`'s job, applied afterward (RB = rho(RB_raw)) by a later
build. `Region.resolved_name`/`.source_type` are always left `None`, and
`CrossReference.resolution`/`.targets`/`.resolves` are always left at
their unresolved defaults, even though the source paragraph and run
styling would in principle let this module compute some of that itself.

The parser reads two kinds of signal the printed page carries structurally
rather than lexically: which runs are italic in the source's own "0Text"
character style (marks a closed-vocabulary POS/register label or an
etymon), and which are styled "1Text" (marks a cross-reference target).
Everything else is closed-vocabulary matching or plain punctuation/
typography pattern matching against the paragraph's concatenated text.

No file IO, no Parquet reads, no model calls. `build_rule_based_context`
does all reusable setup (run grouping, authority-vocabulary lookups) once;
`extract_rule_based_entry` consumes that context per entry.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import pyarrow as pa

from ...schema import (
    Attestation,
    CrossReference,
    Decision,
    Entry,
    Etymology,
    Gloss,
    Label,
    Pronunciation,
    Provenance,
    Region,
    Sense,
    Sourced,
    SourceSpan,
    VariantForm,
)

_DECIDED_BY = "rule_based"

_REQUIRED_RUN_COLUMNS = ("part", "para_idx", "text", "italic", "char_style")
_REQUIRED_SOURCE_FIELDS = ("entry_id", "entry_type", "homonym_index", "para_idx", "raw")
_REQUIRED_AUTHORITY_TABLES = (
    "pos_labels", "register_labels", "language_codes",
    "non_literary_sources", "literary_sources")


class RuleBasedReaderError(RuntimeError):
    """Raised when a rule-based context cannot be built from trustworthy
    input, when a supplied source_entry fails its structural contract, or
    when the parser reaches a state current executable behaviour cannot
    represent reliably (never a fabricated fallback Entry)."""


def _span(para_idx: int) -> SourceSpan:
    return SourceSpan(locator=f"word/document.xml#p{para_idx}", para_idx=para_idx)


# ── grammar/typography patterns (authority-independent) ─────────────────────
# A grammatical-abbreviation shape ("n.", "n. pl.") that the closed
# vocabularies don't happen to declare -- recorded as an extension, never
# silently mapped onto the nearest closed term.
GRAM_LOOKING = re.compile(r"^[a-z]{1,10}\.(\s?[a-z]{1,10}\.)?$")

IPA_OPEN = re.compile(r"^([^/]{0,60}?)\s*/([^/]+)/\s*")
ETYM_MARK = re.compile(r"(?:\bcf\.\s+|<\s*)")
# A closing curly quote is the same character as a contraction/possessive
# apostrophe; only treated as content (not a close) when glued directly
# onto the next word with no space.
QUOTED = re.compile(r"[‘“]((?:[^’”]|’(?=\w)){3,})[’”]")
# A quote immediately glossed by apposition is the dictionary defining the
# quoted phrase itself, not an informant's usage testimony.
MENTION_SIGNAL = re.compile(r"^\s*(?:\([^()]{2,50}\)\s*)?,\s*an?\b|^\s+means\b")
REGION_NEAR = re.compile(r"^\s*\(([A-Z][A-Za-z]{1,5})(?:,\s*[A-Z][a-zA-Zà-ÿ' .-]+)?[,)]")


def _is_mention_quote(text: str, qm: re.Match) -> bool:
    tail = text[qm.end():qm.end() + 40]
    return bool(MENTION_SIGNAL.match(tail)) and not bool(REGION_NEAR.match(tail))


# Group 1's second alternative is deliberately ALL-CAPS-only: an informant
# code is always bare uppercase, while a county name is Title-case, and
# several county names are otherwise short enough to satisfy a looser shape.
REGION_PAREN = re.compile(r"\(([A-Z][A-Za-z]{1,5}(?:,\s*[A-Z]{1,6})?)"
                          r"(?:,\s*([A-Z][a-zA-Zà-ÿ' .-]+?))?(?=[,)])")
# Sense "1." may be the very first thing in the text handed to extract_senses
# (preceded only by a single leftover space), where neither a bare `^`
# anchor nor the usual [.;]-then-whitespace lookbehind has enough context.
SENSE_NUM = re.compile(r"(?:^\s*|(?<=[.;]\s))(\d)\.\s+")

# A variant introduced BEFORE the first pronunciation ("childer also
# childher /.../"), distinct from the after-pronunciation ALSO_VARIANT
# shape below. Excludes "/" from its first group so it never fires on an
# ordinary single-headword entry whose own pronunciation follows.
HEAD_VARIANT_BEFORE = re.compile(
    r"^([^/]+?)\s+(?:also|or)\s+([a-zà-ÿ][\w'’ -]{1,30}?)\s*(?=/)")

QUOTED_ETYM_PAIR = re.compile(r"[‘“]([^’”]{1,60})[’”](?:,\s*([^;.]{1,60}))?")

# "(dimin. of X)" / "(variant of X[, gloss])": the source prints this in
# several shapes -- right after the language code with no separate etymon,
# chained after a real etymon, or free-standing. One pair of regexes
# serves every call site.
DIMIN_OF = re.compile(
    r"dimin\.\s+of\s+([^,).;<]{1,40}?)(?:,\s*([^).;<]{1,60}))?(?=[.,);<]|\s*$)",
    re.IGNORECASE)
VARIANT_OF = re.compile(
    r"variant of\s+([^,).;<]{1,40}?)(?:,\s*([^).;<]{1,60}))?(?=[.,);<]|\s*$)",
    re.IGNORECASE)

# "Author, Title, locus:" or "Author, Title:" immediately before a literary
# quote. Excluding commas from each group individually (not one greedy
# span) keeps the author from being swallowed on a 3-part citation.
LIT_CITATION = re.compile(
    r"([A-ZÁÉÍÓÚ][^,]{1,40}),\s*([^,]{1,60}?)(?:,\s*([^,:]{1,20}))?:\s*$")

# What the source prints instead of a source, right after an oral quote,
# when no informant or work is named ("(proverb)", "(saying)"). Lower-case
# first letter distinguishes this from a region/informant citation.
GENRE_MARKER = re.compile(r"^\s*\(([a-z][a-zà-ÿ' ]{2,60})\)")

# Whole-parenthetical version of REGION_PAREN, for stripping a region
# citation out of definition prose entirely (REGION_PAREN itself uses a
# lookahead instead of capturing the closing ")", so it can't be used with
# .sub()). The caller must still verify the captured code is real.
REGION_PAREN_FULL = re.compile(
    r"\((?P<code>[A-Z][A-Za-z]{1,5})(?:,\s*[A-Z][A-Za-z]{1,5})?"
    r"(?:,\s*[A-Z][a-zA-Zà-ÿ' .-]+?)?[^)]*\)")

# A trailing "See X." redirect pointer, distinct from an inline synonym
# mention -- the literal word "see"/"See" is the signal.
SEE_POINTER = re.compile(r"\.\s+[Ss]ee\s+[A-ZÁÉÍÓÚÜÑ0-9]")

LEADING_QUOTE_ATTRIB = re.compile(
    r"^\s*[‘“]([^’”]{2,200}?)[’”]\s*\(([A-Z][A-Za-z]{1,5})"
    r"(?:,\s*([A-Z][a-zA-Zà-ÿ' .-]+?))?\)")

# A variant printed inside a numbered sense rather than at the entry's own
# head. Requires a trailing comma (the boundary back into ordinary
# definition prose); an optional region paren is re-extracted at sense
# level moments later by the caller, not consumed here.
SENSE_ALSO_VARIANT = re.compile(
    r"^\s*(?:Also|also|or)\s+"
    r"((?:[‘’][^‘’]{1,60}[‘’]|[A-Za-zà-ÿ][\w'-]{0,40})"
    r"(?:,\s*(?:[‘’][^‘’]{1,60}[‘’]|[A-Za-zà-ÿ][\w'-]{0,40}))*)"
    r"\s*(\([A-Z][A-Za-z]{1,5}(?:,\s*[A-Z][a-zA-Zà-ÿ' .-]+)?\))?\s*,\s*")

# Past "see", the whole remainder must be nothing but a target list --
# uppercase/digits/diacritics/separators, no lowercase prose -- so an entry
# that merely mentions "see" mid-definition is never misread as pointer-shaped.
POINTER_SHAPE = re.compile(
    r"^[^,/]{1,50},?\s*[Ss]ee\s+"
    r"[A-ZÁÉÍÓÚÜÑ0-9’'\-][A-ZÁÉÍÓÚÜÑ0-9’'\-.,;:() ]*\.?\s*$")


def _italic_mask(runs: Sequence[Mapping[str, object]]) -> list[bool]:
    """One bool per character of the concatenated run text, so a text-level
    regex match can be checked against styling before being trusted."""
    mask: list[bool] = []
    for r in runs:
        mask.extend([bool(r["italic"])] * len(r["text"]))
    return mask


def _is_mostly_italic(mask: list[bool], start: int, end: int) -> bool:
    span_mask = mask[start:end]
    if not span_mask:
        return False
    return sum(span_mask) / len(span_mask) > 0.6


def _split_variant_blob(blob: str) -> tuple[list[str], str | None]:
    """Split a captured "X, Y, Z" or "X, Y (CODE, County)" variant blob into
    individual names. An embedded region parenthetical is pulled out
    separately -- it sits before the offset region extraction later scans
    from, so left in place it would be silently lost, not just mis-parsed."""
    region_m = REGION_PAREN_FULL.search(blob)
    region_text = None
    if region_m:
        region_text = blob[region_m.start():region_m.end()]
        blob = blob[:region_m.start()] + blob[region_m.end():]
    blob = re.sub(r"\s{2,}", " ", blob).strip().strip(",")
    parts = re.split(r",\s*|\s+sometimes\s+", blob)
    variants = [re.sub(r"^sometimes\s+", "", p.strip()).strip("’'").rstrip(".")
               for p in parts if p.strip()]
    return variants, region_text


# ── run/authority context ───────────────────────────────────────────────────
@dataclass(frozen=True)
class RuleBasedContext:
    """Precomputed, reusable state for `extract_rule_based_entry`: the
    document body's runs grouped by paragraph, and closed-vocabulary
    lookups derived from caller-supplied apparatus tables. Built once by
    `build_rule_based_context` and safe to reuse across an entire corpus --
    `extract_rule_based_entry` never rebuilds any of it or reads a file.
    """

    paragraphs: Mapping[int, tuple[Mapping[str, object], ...]]
    pos_terms: frozenset[str]
    register_terms: frozenset[str]
    language_terms: frozenset[str]
    informant_rows_by_code: Mapping[str, tuple[Mapping[str, object], ...]]
    known_informant_codes: frozenset[str]
    county_names: frozenset[str]
    literary_author_surnames: frozenset[str]
    also_variant_pattern: re.Pattern[str]


def _validate_runs_schema(runs: pa.Table) -> None:
    names = runs.schema.names
    if len(set(names)) != len(names):
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        raise RuleBasedReaderError(f"runs table has duplicate column name(s): {dup}")
    missing = [c for c in _REQUIRED_RUN_COLUMNS if c not in names]
    if missing:
        raise RuleBasedReaderError(f"runs table is missing required column(s): {missing}")


def _validate_run_rows(rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        if not isinstance(r["para_idx"], int) or isinstance(r["para_idx"], bool):
            raise RuleBasedReaderError(
                f"run row {i}: para_idx must be a plain int, got {r['para_idx']!r}")
        if not isinstance(r["text"], str):
            raise RuleBasedReaderError(
                f"run row {i} (para_idx={r['para_idx']!r}): text must be a "
                f"string, got {type(r['text']).__name__}")


def _validate_authority_schema(table: pa.Table, name: str, required: tuple[str, ...]) -> None:
    names = table.schema.names
    if len(set(names)) != len(names):
        dup = sorted(n for n, c in Counter(names).items() if c > 1)
        raise RuleBasedReaderError(
            f"authority table {name!r} has duplicate column name(s): {dup}")
    missing = [c for c in required if c not in names]
    if missing:
        raise RuleBasedReaderError(
            f"authority table {name!r} is missing required column(s): {missing}")


def _term_set(table: pa.Table, name: str) -> frozenset[str]:
    _validate_authority_schema(table, name, ("term",))
    terms = set()
    for row in table.to_pylist():
        term = row.get("term")
        if not term:
            raise RuleBasedReaderError(f"authority table {name!r} has an empty 'term'")
        terms.add(term)
    return frozenset(terms)


def build_rule_based_context(
    runs: pa.Table,
    authority_tables: Mapping[str, pa.Table],
) -> RuleBasedContext:
    """Build a reusable `RuleBasedContext` from the public ingestion run
    table and caller-supplied apparatus tables.

    Only `part == "word/document.xml"` rows are grouped, by `para_idx`,
    preserving row order -- other parts (headers, footnotes) have their
    own independent paragraph numbering and are excluded entirely, never
    silently merged in.

    Requires `pos_labels`, `register_labels`, `language_codes`,
    `non_literary_sources`, `literary_sources` in `authority_tables`
    (`bibliography` is not read: the current parser's own region
    extraction never reaches informant resolution's bibliography-fallback
    branch, since it only calls into it for a code already confirmed
    present in `non_literary_sources`).

    Raises `RuleBasedReaderError` on a missing/duplicate column, a
    malformed run row, a missing authority table, or an authority row
    with an empty required field -- never silently drops a malformed row.
    """
    _validate_runs_schema(runs)
    body_rows = [r for r in runs.to_pylist() if r["part"] == "word/document.xml"]
    _validate_run_rows(body_rows)

    paragraphs: dict[int, list[dict]] = defaultdict(list)
    for r in body_rows:
        paragraphs[r["para_idx"]].append(r)

    missing_tables = [t for t in _REQUIRED_AUTHORITY_TABLES if t not in authority_tables]
    if missing_tables:
        raise RuleBasedReaderError(
            f"authority_tables is missing required table(s): {missing_tables}")

    pos_terms = _term_set(authority_tables["pos_labels"], "pos_labels")
    register_terms = _term_set(authority_tables["register_labels"], "register_labels")
    language_terms = _term_set(authority_tables["language_codes"], "language_codes")

    non_literary = authority_tables["non_literary_sources"]
    _validate_authority_schema(non_literary, "non_literary_sources", ("code", "county"))
    informant_rows_by_code: dict[str, list[dict]] = defaultdict(list)
    # `county_names` reproduces current behaviour's own code->row collapse
    # (last-row-per-code wins on a shared code) exactly, not a plain set of
    # every printed county: a handful of counties are only ever printed on
    # a row whose code is shared with a later row, and current behaviour's
    # bare-locality check never sees them. This is a known upstream defect
    # (see hiberno_english.resolution's own county-collision handling for
    # the corrected version), reproduced here on purpose for behavioural
    # equivalence -- `informant_rows_by_code` below stays collision-aware.
    last_row_per_code: dict[str, dict] = {}
    for row in non_literary.to_pylist():
        code = row.get("code")
        if not code:
            raise RuleBasedReaderError("non_literary_sources row has an empty 'code'")
        informant_rows_by_code[code].append(row)
        last_row_per_code[code] = row
    county_names = {row["county"] for row in last_row_per_code.values() if row.get("county")}

    literary = authority_tables["literary_sources"]
    _validate_authority_schema(literary, "literary_sources", ("author",))
    literary_author_surnames = frozenset(
        row["author"].split(",")[0].strip()
        for row in literary.to_pylist() if row.get("author"))

    also_variant_pattern = re.compile(
        r"^(?:also|or)\s+([a-zà-ÿ][\w'’(),.\- ]{1,120}?)\s+(?=" +
        "|".join(re.escape(t) for t in sorted(pos_terms, key=len, reverse=True)) +
        r")")

    return RuleBasedContext(
        paragraphs=MappingProxyType({k: tuple(v) for k, v in paragraphs.items()}),
        pos_terms=pos_terms,
        register_terms=register_terms,
        language_terms=language_terms,
        informant_rows_by_code=MappingProxyType(
            {k: tuple(v) for k, v in informant_rows_by_code.items()}),
        known_informant_codes=frozenset(informant_rows_by_code),
        county_names=frozenset(county_names),
        literary_author_surnames=literary_author_surnames,
        also_variant_pattern=also_variant_pattern,
    )


# ── field-group 1: headword, pronunciation, variant forms ──────────────────
def _extract_head(
    text: str, runs: Sequence[Mapping[str, object]], context: RuleBasedContext,
) -> tuple[str, list[str], list[str], int, str | None]:
    """Returns (headword, ipa_list, variant_forms, text_offset_after_head,
    embedded_region_paren)."""
    m = IPA_OPEN.match(text)
    if not m:
        # No printed IPA pronunciation at all -- fall back to the same
        # italic-0Text-run signal label extraction itself uses: everything
        # before the first such run is the headword. Only a plain
        # first-comma split if the entry has no italic 0Text run anywhere.
        pos = 0
        for r in runs:
            if r["italic"] and r["char_style"] == "0Text":
                break
            pos += len(r["text"])
        else:
            pos = 0
        head = text[:pos].strip().rstrip(",") if pos > 0 else text.split(",", 1)[0].strip()
        return head, [], [], pos, None

    variants: list[str] = []
    headword = m.group(1).strip()
    bm = HEAD_VARIANT_BEFORE.match(text)
    if bm:
        headword = bm.group(1).strip()
        variants.append(bm.group(2).strip())

    # Every consecutive /pron/ block at the head counts, not just the first.
    ipa_list = [m.group(2).strip()]
    pos = m.end()
    while True:
        nm = re.match(r"^,?\s*/([^/]+)/\s*", text[pos:])
        if not nm:
            break
        ipa_list.append(nm.group(1).strip())
        pos += nm.end()

    rest = text[pos:]
    vm = context.also_variant_pattern.match(rest)
    embedded_region = None
    if vm:
        var_list, embedded_region = _split_variant_blob(vm.group(1))
        variants.extend(var_list)
        rest_offset = pos + vm.end()
    else:
        rest_offset = pos
    return headword, ipa_list, variants, rest_offset, embedded_region


# ── field-group 2: POS / register labels ────────────────────────────────────
def _extract_labels(
    runs: Sequence[Mapping[str, object]], offset: int, context: RuleBasedContext,
) -> tuple[list[Label], int, list[str]]:
    """Walks italic 0Text runs starting at `offset`, tokenising on
    whitespace within each run. A trailing italic run's period sometimes
    lands in the following roman run -- merged back on before matching.
    Stops at the first run carrying real (non label-shaped) content."""
    labels: list[Label] = []
    extensions: list[str] = []
    pos = 0
    i = 0
    stopped_at = offset
    n = len(runs)
    while i < n:
        r = runs[i]
        start, end = pos, pos + len(r["text"])
        pos = end
        if end <= offset:
            i += 1
            continue
        seg = r["text"][max(0, offset - start):]
        if not seg.strip():
            i += 1
            stopped_at = pos
            continue
        if not (r["italic"] and r["char_style"] == "0Text"):
            break
        tokens = seg.split()
        nxt = runs[i + 1]["text"] if i + 1 < n else ""
        if tokens and nxt.startswith(".") and not tokens[-1].endswith("."):
            tokens[-1] += "."
            stopped_at = pos + 1
        else:
            stopped_at = pos
        for tok in tokens:
            # A register term is sometimes parenthesised on the POS run
            # itself ("p.part. (colloq.),") -- stripped before matching.
            tok = tok.strip(",").strip("()").strip(",")
            if not tok:
                continue
            if tok in context.pos_terms:
                labels.append(Label(
                    term=Sourced(value=tok, authority_ref=f"pos_labels:{tok}", decided_by=_DECIDED_BY),
                    kind="pos"))
            elif tok in context.register_terms:
                labels.append(Label(
                    term=Sourced(value=tok, authority_ref=f"register_labels:{tok}", decided_by=_DECIDED_BY),
                    kind="register"))
            elif GRAM_LOOKING.match(tok):
                extensions.append(tok)
            else:
                # Not grammatical-abbreviation-shaped -- this italic run is
                # doing something else. Stop the label scan rather than
                # swallow prose.
                return labels, offset, extensions
        i += 1
    return labels, stopped_at, extensions


# ── field-group 3: etymology ─────────────────────────────────────────────────
def _component_pieces(region: str) -> list[str] | None:
    """Split a "ETYM1[, gloss1] + ETYM2[, gloss2] [+ ...]" component blob on
    its own depth-0 "+" markers. Paren-depth aware, since a piece's own
    gloss can itself be parenthesised. Stops at the first depth-0 sentence
    end, semicolon, quote, or a second "<" marker. Returns None (not a
    component blob) when no depth-0 "+" is found."""
    depth = 0
    end = len(region)
    plus_positions = []
    for i, ch in enumerate(region):
        if ch in "‘“":
            end = i
            break
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and ch == "<":
            end = i
            break
        elif depth == 0 and ch == ".":
            nxt = region[i + 1:i + 2]
            after_ws = region[i + 1:].lstrip(" \t")[:1]
            if nxt == "" or (nxt in " \t" and (after_ws == "" or after_ws.isupper())):
                end = i
                break
        elif depth == 0 and ch == ";":
            end = i
            break
        elif depth == 0 and ch == "+":
            plus_positions.append(i)
    if not plus_positions:
        return None
    bounds = [0] + [p + 1 for p in plus_positions]
    ends = plus_positions + [end]
    return [region[b:e].strip() for b, e in zip(bounds, ends)]


def _parse_component_piece(piece: str) -> tuple[str, str | None]:
    """One "+"-joined element's own form and optional gloss, marked either
    by a parenthetical or a trailing comma (tried in that order, since a
    comma can legitimately appear inside a parenthesised gloss)."""
    piece = piece.strip().rstrip(".,;")
    pm = re.search(r"\(([^()]{1,60})\)", piece)
    if pm:
        return piece[:pm.start()].strip(), pm.group(1).strip()
    cm = re.match(r"^([^,]{1,40}?),\s*(.{1,60})$", piece)
    if cm:
        return cm.group(1).strip(), cm.group(2).strip()
    return piece, None


def _extract_etymology(
    runs: Sequence[Mapping[str, object]], text: str, context: RuleBasedContext,
) -> tuple[list[Etymology], list[str]]:
    mask = _italic_mask(runs)
    out: list[Etymology] = []
    extensions: list[str] = []
    last_lang: str | None = None
    consumed_dimin_variant: list[tuple[int, int]] = []

    for m in ETYM_MARK.finditer(text):
        is_cf = text[m.start():m.end()].strip().startswith("cf")
        rest = text[m.end():].lstrip()
        # A parenthetical qualifier before the language code itself
        # ("< (O)F vexer") starts with "(" rather than a letter.
        qm = re.match(r"\([A-Za-z]{1,10}\)\s*", rest)
        qualifier_len = qm.end() if qm else 0
        lm = re.match(r"([A-Za-z][\w. ]{0,10}?)(?=\s|$)", rest[qualifier_len:])
        if not lm:
            continue
        lang_full = lm.group(1)
        lang = next((c for c in (lang_full, lang_full.rstrip(".")) if c in context.language_terms), None)
        if lang is None:
            # A real language attribution the apparatus doesn't declare:
            # require it to look like an actual language name AND have
            # real content following, the same signal a genuine etymon
            # needs -- keeps this from logging every stray capitalised word.
            tok = lang_full.rstrip(".")
            if re.match(r"^[A-Z][a-z]{3,}$", tok):
                probe_pos = m.end() + qualifier_len + lm.end()
                while probe_pos < len(text) and text[probe_pos] == " ":
                    probe_pos += 1
                has_content = (probe_pos < len(text) and
                              (text[probe_pos] in "‘“" or
                               (probe_pos < len(mask) and mask[probe_pos])))
                if has_content:
                    extensions.append(tok)
            continue
        relation = "comparison" if is_cf else "derivation"
        last_lang = lang
        etymon_start = m.end() + qualifier_len + lm.end()
        etymon_region = text[etymon_start:etymon_start + 200]

        stripped = etymon_region.lstrip()
        lead_ws = len(etymon_region) - len(stripped)

        # A parenthetical directly replacing the etymon: no separate italic
        # etymon is printed at all, and the marker's own remaining content
        # IS the fact.
        if stripped[:1] == "(":
            paren_probe = stripped[1:]
            dm = DIMIN_OF.match(paren_probe)
            vm = None if dm else VARIANT_OF.match(paren_probe)
            hit = dm or vm
            if hit:
                out.append(Etymology(
                    language_code=Sourced(value=lang, authority_ref=f"language_codes:{lang}", decided_by=_DECIDED_BY),
                    etymon_form=hit.group(1).strip(),
                    gloss=(hit.group(2).strip() if hit.group(2) else None),
                    relation="diminutive_of" if dm else "variant_of",
                ))
                consumed_dimin_variant.append(
                    (etymon_start + lead_ws + 1, etymon_start + lead_ws + 1 + hit.end()))
                continue

        if stripped[:1] in "‘“":
            # One or more quoted phrases, semicolon-separated, each with
            # its own optional gloss -- stop at the first quote-less clause.
            pos = 0
            for pm in QUOTED_ETYM_PAIR.finditer(stripped):
                if pm.start() != pos and stripped[pos:pm.start()].strip(" ;"):
                    break
                abs_start = etymon_start + lead_ws + pm.start(1)
                abs_end = abs_start + len(pm.group(1))
                if not _is_mostly_italic(mask, abs_start, abs_end):
                    # A bare language marker with no etymon actually given
                    # -- the next quote is a roman example sentence.
                    break
                out.append(Etymology(
                    language_code=Sourced(value=lang, authority_ref=f"language_codes:{lang}", decided_by=_DECIDED_BY),
                    etymon_form=pm.group(1).strip().rstrip(",."),
                    gloss=(pm.group(2).strip() if pm.group(2) else None),
                    relation=relation,
                ))
                pos = pm.end()
                if not stripped[pos:pos + 2].strip().startswith(";"):
                    break
        else:
            # A component construction: two or more elements combining to
            # form the word, not a chain of successive ancestors.
            pieces = _component_pieces(stripped)
            if pieces and len(pieces) > 1:
                for piece in pieces:
                    form, piece_gloss = _parse_component_piece(piece)
                    if not form:
                        continue
                    out.append(Etymology(
                        language_code=Sourced(value=lang, authority_ref=f"language_codes:{lang}", decided_by=_DECIDED_BY),
                        etymon_form=form, gloss=piece_gloss, relation="component",
                    ))
                continue
            # The italic run boundary marks the etymon directly, rather
            # than a punctuation-class regex guessing where prose resumes.
            pos = etymon_start
            while pos < len(text) and text[pos] == " ":
                pos += 1
            span_start = pos
            while pos < len(mask) and mask[pos]:
                pos += 1
            etymon_form = text[span_start:pos].strip().rstrip(",.);")
            # "See" is italicised the same way a genuine etymon is, but is
            # always an editorial redirect marker, never part of the etymon.
            see_m = re.search(r"(?:^|\.\s+)[Ss]ee\b", etymon_form)
            if see_m:
                etymon_form = etymon_form[:see_m.start()].strip().rstrip(",.")
            if not etymon_form:
                continue
            gloss = None
            if not see_m:
                tail = text[pos:pos + 90]
                gm = re.match(r",?\s*([^.;()<‘’“”]{2,60}?)(?=[.;()<‘’“”]|\s*$)", tail)
                if gm and gm.group(1).strip():
                    gloss = gm.group(1).strip()
                    gsee_m = re.search(r"\bSee\b", gloss)
                    if gsee_m:
                        gloss = gloss[:gsee_m.start()].strip().rstrip(",.;’“") or None
                    if gloss and re.fullmatch(r"cf\.?", gloss, re.IGNORECASE):
                        gloss = None
                # A further diminutive/variant fact chained after this
                # etymon -- one level further back (chain_step 2).
                cdm = DIMIN_OF.search(tail)
                cvm = None if cdm else VARIANT_OF.search(tail)
                chained = cdm or cvm
                if chained and gloss and chained.group(0) in gloss:
                    gloss = None
                if chained:
                    out.append(Etymology(
                        language_code=Sourced(value=lang, authority_ref=f"language_codes:{lang}", decided_by=_DECIDED_BY),
                        etymon_form=chained.group(1).strip(),
                        gloss=(chained.group(2).strip() if chained.group(2) else None),
                        relation="diminutive_of" if cdm else "variant_of",
                        chain_step=2,
                    ))
                    consumed_dimin_variant.append((pos + chained.start(), pos + chained.end()))
            out.append(Etymology(
                language_code=Sourced(value=lang, authority_ref=f"language_codes:{lang}", decided_by=_DECIDED_BY),
                etymon_form=etymon_form,
                gloss=gloss,
                relation=relation,
            ))

    # A standalone diminutive/variant fact with no parenthetical and no
    # adjacency to any etymon run at all -- swept once over the whole
    # entry, attached only when an unambiguous language is available.
    if last_lang is not None:
        for rx, rel in ((DIMIN_OF, "diminutive_of"), (VARIANT_OF, "variant_of")):
            for bm in rx.finditer(text):
                if any(s <= bm.start() < e for s, e in consumed_dimin_variant):
                    continue
                out.append(Etymology(
                    language_code=Sourced(value=last_lang, authority_ref=f"language_codes:{last_lang}", decided_by=_DECIDED_BY),
                    etymon_form=bm.group(1).strip(),
                    gloss=(bm.group(2).strip() if bm.group(2) else None),
                    relation=rel,
                ))
    return out, extensions


# ── field-group 4: regions, attestations ─────────────────────────────────────
def _extract_regions(text: str, context: RuleBasedContext) -> list[Region]:
    """Extracts `informant_code`/`county` only -- `resolved_name`/
    `source_type` are always left `None`; identity resolution is rho's job.

    `county` may still be backfilled from a uniquely-identified authority
    row's own printed county when a citation names a code but no county at
    all: this is a plain, unambiguous lookup on one already-required
    authority table (never a name/source_type), and reproduces current
    executable behaviour exactly for this field. A SHARED code with no
    printed county is never backfilled -- ambiguity is never guessed.
    """
    out = []
    for m in REGION_PAREN.finditer(text):
        code = m.group(1)
        base_code = code.split(",")[0].strip()
        if base_code not in context.known_informant_codes:
            # A bare locality with no informant code at all -- only when
            # nothing else is printed alongside it.
            if base_code in context.county_names and not m.group(2):
                out.append(Region(informant_code=None, county=base_code))
            continue
        printed_county = m.group(2).strip() if m.group(2) else None
        county = printed_county
        if not county:
            rows = context.informant_rows_by_code.get(base_code, ())
            if len(rows) == 1:
                county = rows[0].get("county")
        out.append(Region(
            informant_code=Sourced(value=base_code, authority_ref=f"non_literary_sources:{base_code}",
                                   decided_by=_DECIDED_BY),
            county=county,
        ))
    return out


def _resolve_author(short: str, literary_author_surnames: frozenset[str]) -> str:
    for surname in literary_author_surnames:
        if surname.split(",")[0].strip() == short.strip():
            return surname
    return short


def _extract_attestations(text: str, context: RuleBasedContext) -> list[Attestation]:
    out = []
    for m in QUOTED.finditer(text):
        quote = m.group(1)
        opening = text[m.start()]
        # Double curly quotes = literary, single = oral (the source's own
        # citation convention). Neither -> not a real citation.
        citation_class = "literary" if opening == "“" else "oral" if opening == "‘" else None
        if citation_class is None:
            continue
        if _is_mention_quote(text, m):
            continue
        if citation_class == "oral":
            tail = text[m.end():m.end() + 90]
            rm = REGION_PAREN.search(tail)
            if not rm:
                # "(CODE, County, who adds: 'quote')" -- the region wraps
                # and precedes the quote here. Only taken if the quote
                # sits inside the same still-open parenthesis.
                before = text[:m.start()]
                if before.count("(") > before.count(")"):
                    rm = REGION_PAREN.search(before[max(0, len(before) - 90):])
            if not rm:
                gm = GENRE_MARKER.match(tail)
                if gm:
                    out.append(Attestation(quote=quote, citation_class="oral",
                                           source="", genre=gm.group(1).strip()))
                continue
            code = rm.group(1).split(",")[0].strip()
            source = f"{code}, {rm.group(2)}" if rm.group(2) else code
            out.append(Attestation(quote=quote, citation_class="oral", source=source))
        else:
            before = text[max(0, m.start() - 120):m.start()]
            # Restricted to the last sentence of the window, so the
            # author group never swallows leftover text from an earlier
            # sentence.
            last_boundary = max(before.rfind(". "), before.rfind("; "),
                                before.rfind(") "))
            before = before[last_boundary + 2:] if last_boundary >= 0 else before
            am = LIT_CITATION.search(before)
            if not am:
                continue
            author = _resolve_author(am.group(1).strip(), context.literary_author_surnames)
            source = f"{author}, {am.group(2).strip()}"
            out.append(Attestation(quote=quote, citation_class="literary",
                                   source=source,
                                   locus=am.group(3).strip() if am.group(3) else None))
    return out


# ── field-group 5: cross-references ──────────────────────────────────────────
def _extract_cross_refs(
    runs: Sequence[Mapping[str, object]], is_pointer_entry: bool,
) -> list[CrossReference]:
    """A 1Text run marks a cross-reference target. Consecutive 1Text runs
    joined only by a bare, unspaced intra-word connector ("(", ")", "-")
    are one target, not several. Raw extraction only -- `resolution`/
    `targets`/`resolves` are left at their unresolved defaults; rho fills
    them in afterward."""
    out = []
    i, n = 0, len(runs)
    while i < n:
        t = runs[i]["text"].strip()
        # A target needs 2+ consecutive uppercase letters, or a single
        # uppercase letter directly followed by a digit (catches
        # postal-code-style slang headwords like "D4").
        if runs[i]["char_style"] == "1Text" and re.search(r"[A-ZÁÉÍÓÚÜÑ]{2,}|[A-ZÁÉÍÓÚÜÑ]\d", t):
            pieces = [runs[i]["text"]]
            j = i + 1
            while (j + 1 < n and runs[j]["char_style"] != "1Text"
                   and runs[j]["text"] in ("(", ")", "-")
                   and runs[j + 1]["char_style"] == "1Text"
                   and re.search(r"[A-ZÁÉÍÓÚÜÑ]", runs[j + 1]["text"])):
                pieces.append(runs[j]["text"])
                pieces.append(runs[j + 1]["text"])
                j += 2
            # Leading open-quote marks are stripped; a leading apostrophe
            # or hyphen is not, since both are real leading characters
            # some targets have.
            joined = "".join(pieces).strip().lstrip("‘“").rstrip(".,;)”")
            # A "See A; B; C." list sometimes prints as one unbroken
            # 1Text run with no run boundary at the semicolons.
            candidates = ([p.strip() for p in joined.split(";")]
                         if "; " in joined else [joined])
            for target in candidates:
                target = target.strip().rstrip(".,;)”")
                # 1Text also marks pure typographic emphasis inside a
                # quotation, not only cross-references. A genuine target is
                # always a short bare headword/phrase.
                looks_like_target = (target and not target.isdigit()
                                     and len(target) <= 40
                                     and "[" not in target and "‘" not in target)
                if looks_like_target:
                    out.append(CrossReference(
                        target=Sourced(value=target, decided_by=_DECIDED_BY),
                        type="redirect" if is_pointer_entry else "inline",
                    ))
            i = j
        else:
            i += 1
    return out


# ── definitions, senses ──────────────────────────────────────────────────────
def _find_definition_end(text: str, known_codes: frozenset[str]) -> int:
    """Where definition prose ends and etymology/attestation content
    begins -- the min of the first etymology marker, the first (non-
    mention) quoted span, and a trailing "See X." pointer.

    A quote sitting inside a still-open "(CODE, County, who adds: '...')"
    parenthetical cuts at the parenthetical's own open paren instead, but
    only when that paren is a real known region code.
    """
    candidates = []
    em = ETYM_MARK.search(text)
    if em:
        candidates.append(em.start())
    qm = None
    for cand in QUOTED.finditer(text):
        if not _is_mention_quote(text, cand):
            qm = cand
            break
    if qm:
        before = text[:qm.start()]
        open_paren = before.rfind("(")
        close_paren = before.rfind(")")
        if open_paren > close_paren:
            pm = REGION_PAREN_FULL.match(before, open_paren)
            if pm and pm.group("code") in known_codes:
                candidates.append(open_paren)
            else:
                candidates.append(qm.start())
        else:
            candidates.append(qm.start())
    sm = SEE_POINTER.search(text)
    if sm:
        candidates.append(sm.start())
    return min(candidates) if candidates else len(text)


def _clean_definition(raw: str, known_codes: frozenset[str]) -> str:
    def _strip_if_real(m: re.Match) -> str:
        return "" if m.group("code") in known_codes else m.group(0)

    d = REGION_PAREN_FULL.sub(_strip_if_real, raw)
    d = re.sub(r"\s+,", ",", d)
    d = re.sub(r"\s+;", ";", d)
    d = re.sub(r"\s+\.", ".", d)
    d = re.sub(r",\s*,", ",", d)
    d = re.sub(r"\s{2,}", " ", d)
    # A trailing unmatched "(" survives when the cut point sits inside a
    # parenthetical aside rather than starting it -- never legitimate. A
    # genuine leading "(" opening a definition is left alone, so only the
    # trailing case gets its own rstrip.
    return d.strip().strip(",;").rstrip("(").strip()


def _leading_definition_attribution(
    chunk: str, known_codes: frozenset[str],
) -> tuple[str | None, str | None]:
    """(attribution, exact quote text) for a leading quote-plus-citation
    shape ("'a definition' (CODE, County)"), or (None, None). The quote
    text travels back too so the caller can keep it from also being
    recorded as an (incorrect) oral attestation."""
    m = LEADING_QUOTE_ATTRIB.match(chunk)
    if not m:
        return None, None
    code = m.group(2)
    if code not in known_codes:
        return None, None
    attribution = f"{code}, {m.group(3)}" if m.group(3) else code
    return attribution, m.group(1)


def _extract_senses(
    text: str, runs: Sequence[Mapping[str, object]], context: RuleBasedContext,
    base_offset: int = 0,
) -> tuple[list[Sense], bool, list[str], list[VariantForm]]:
    """Returns (senses, per_sense_pos, pos_extensions, sense_variants).

    `per_sense_pos` is True when the source prints part of speech again
    after EACH numbered sense marker, rather than once before sense 1 --
    see `Entry.pos`'s own schema docstring for why the two conventions
    must not be conflated. `sense_variants` collects variant forms printed
    inside a numbered sense, for the caller to fold into
    `Entry.variant_forms`.
    """
    nums = list(SENSE_NUM.finditer(text))
    if not nums:
        chunks = [(None, 0, len(text))]
    else:
        chunks = []
        for i, m in enumerate(nums):
            start = m.end()
            end = nums[i + 1].start() if i + 1 < len(nums) else len(text)
            chunks.append((int(m.group(1)), start, end))

    # An italic run immediately after a "N. " marker that is label-shaped
    # is the same signal label extraction itself uses. Only engaged when
    # 2+ numbered senses exist AND at least one shows this signal directly.
    per_sense_info: dict[int, tuple[list[Label], int]] = {}
    per_sense_extensions: list[str] = []
    if len(nums) >= 2:
        for number, start, _end in chunks:
            lbls, new_off, ext = _extract_labels(runs, base_offset + start, context)
            if lbls:
                per_sense_info[number] = (lbls, new_off)
                per_sense_extensions.extend(ext)
    per_sense_pos = bool(per_sense_info)

    known_codes = context.known_informant_codes
    sense_variants: list[VariantForm] = []
    senses = []
    for number, start, end in chunks:
        chunk_labels: list[Label] = []
        if per_sense_pos and number in per_sense_info:
            chunk_labels, new_off = per_sense_info[number]
            start = new_off - base_offset
        chunk = text[start:end]
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            continue

        # A variant printed at the very start of THIS numbered sense,
        # consumed here so it doesn't leak into the definition text below.
        # Region/attestation extraction below still runs on the full
        # chunk, so the same citation is still attached at sense level too.
        def_start = 0
        vm = SENSE_ALSO_VARIANT.match(chunk_stripped)
        if vm and number is not None:
            var_regions = _extract_regions(vm.group(2), context) if vm.group(2) else []
            for raw_v in re.split(r",\s*", vm.group(1)):
                raw_v = raw_v.strip().strip("‘’").strip()
                if raw_v:
                    sense_variants.append(VariantForm(
                        form=Sourced(value=raw_v, decided_by=_DECIDED_BY),
                        regions=var_regions, sense_number=number))
            def_start = vm.end()
        post_variant = chunk_stripped[def_start:]

        def_end = _find_definition_end(post_variant, known_codes)
        raw_def_slice = post_variant[:def_end]
        definition = _clean_definition(raw_def_slice, known_codes)
        if not definition:
            # The chunk opens straight into a quote/etymology marker with
            # no prose of its own. Still cut the trailing pointer at
            # minimum; the quote itself stays, since it's all there is.
            sm = SEE_POINTER.search(post_variant)
            raw_def_slice = post_variant[:sm.start()] if sm else post_variant
            definition = _clean_definition(raw_def_slice, known_codes)

        definition_attribution, attrib_quote = _leading_definition_attribution(
            post_variant, known_codes)

        attestations = _extract_attestations(chunk_stripped, context)
        if attrib_quote:
            attestations = [a for a in attestations
                            if not (a.quote == attrib_quote and a.source == definition_attribution)]
        if vm and number is not None:
            # A variant blob can itself contain quote-styled names that
            # attestation extraction -- run moments earlier on the
            # untrimmed chunk -- reads as oral attestation quotes.
            variant_forms_here = {sv.form.value for sv in sense_variants
                                  if sv.sense_number == number}
            attestations = [a for a in attestations if a.quote not in variant_forms_here]

        # A per-clause split, additive to `definition` (which stays the
        # full verbatim run). Only on a semicolon split, and only kept
        # when at least one piece carries its own region citation -- the
        # actual signal that per-clause attribution is happening at all.
        glosses: list[Gloss] = []
        pieces = re.split(r";\s*", raw_def_slice)
        if len(pieces) >= 2:
            piece_glosses = []
            any_region = False
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                piece_regions = _extract_regions(piece, context)
                if piece_regions:
                    any_region = True
                piece_text = _clean_definition(piece, known_codes)
                if piece_text:
                    piece_glosses.append(Gloss(
                        text=Sourced(value=piece_text, decided_by=_DECIDED_BY),
                        regions=piece_regions))
            if any_region:
                glosses = piece_glosses

        senses.append(Sense(
            number=number,
            definition=Sourced(value=definition, decided_by=_DECIDED_BY),
            definition_attribution=definition_attribution,
            glosses=glosses,
            labels=chunk_labels,
            regions=_extract_regions(chunk_stripped, context),
            attestations=attestations,
        ))
    return senses, per_sense_pos, per_sense_extensions, sense_variants


# ── top-level entry assembly ─────────────────────────────────────────────────
def _validate_source_entry(source_entry: Mapping[str, object]) -> None:
    missing = [f for f in _REQUIRED_SOURCE_FIELDS if f not in source_entry]
    if missing:
        raise RuleBasedReaderError(f"source_entry is missing required field(s): {missing}")

    entry_id = source_entry["entry_id"]
    if not isinstance(entry_id, str) or not entry_id:
        raise RuleBasedReaderError(
            f"source_entry.entry_id must be a non-empty string, got {entry_id!r}")

    entry_type = source_entry["entry_type"]
    if not isinstance(entry_type, str) or not entry_type:
        raise RuleBasedReaderError(
            f"source_entry.entry_type must be a non-empty string, got {entry_type!r}")
    if entry_type == "letter_header":
        raise RuleBasedReaderError(
            f"source_entry {entry_id!r} is a letter_header, not a lexical entry")

    homonym_index = source_entry["homonym_index"]
    if homonym_index is not None and (
           isinstance(homonym_index, bool) or not isinstance(homonym_index, int)):
        raise RuleBasedReaderError(
            f"source_entry.homonym_index must be None or a plain int, got {homonym_index!r}")

    para_idx = source_entry["para_idx"]
    if isinstance(para_idx, bool) or not isinstance(para_idx, int):
        raise RuleBasedReaderError(
            f"source_entry.para_idx must be a plain int, got {para_idx!r}")

    raw = source_entry["raw"]
    if not isinstance(raw, str):
        raise RuleBasedReaderError(
            f"source_entry.raw must be a string, got {type(raw).__name__}")


def extract_rule_based_entry(
    source_entry: Mapping[str, object],
    context: RuleBasedContext,
    *,
    provenance: Provenance,
) -> Entry:
    """Deterministically extract one unresolved lexical `Entry` (RB_raw)
    from `source_entry` (one row of public segmentation output) and its
    paragraph's run styling in `context`.

    `provenance` is deep-copied before use; `Provenance.source_span` comes
    from the caller exactly as supplied. A per-leaf `SourceSpan` derived
    from `source_entry`'s own `para_idx` (never a page number, never a
    witness filename) is additionally attached to the headword, matching
    current executable behaviour's own scope for that field.

    Fills `Region.informant_code`/`.county` and `CrossReference.target`/
    `.type` only -- `Region.resolved_name`/`.source_type` and
    `CrossReference.resolution`/`.targets`/`.resolves` are always left at
    their unresolved defaults. `Entry.usage_notes` and `.sub_entries` stay
    at their empty defaults, matching current executable behaviour, which
    extracts neither.

    Raises `RuleBasedReaderError` on a structural precondition failure
    (malformed `source_entry`, no run context for its paragraph) or when
    the parser reaches a state current executable behaviour cannot
    represent reliably (a pointer-shaped entry with no cross-reference
    target recoverable at all) -- never a fabricated fallback `Entry`.
    Ordinary lexical abstention (no etymology, no cross-reference, no
    usage note) remains valid and yields an empty list, not an error.
    """
    _validate_source_entry(source_entry)

    entry_id = source_entry["entry_id"]
    entry_type = source_entry["entry_type"]
    homonym_index = source_entry["homonym_index"]
    para_idx = source_entry["para_idx"]
    text = source_entry["raw"]

    if para_idx not in context.paragraphs:
        raise RuleBasedReaderError(
            f"no run context for paragraph {para_idx} (entry_id={entry_id!r})")
    runs = context.paragraphs[para_idx]

    prov = provenance.model_copy(deep=True)
    is_pointer = entry_type == "cross_ref" or bool(POINTER_SHAPE.match(text))

    if is_pointer:
        head = text.split(",")[0].strip()
        if homonym_index is not None and head.endswith(str(homonym_index)):
            head = head[:-len(str(homonym_index))].strip()
        xrefs = _extract_cross_refs(runs, is_pointer_entry=True)
        if not xrefs:
            # A documented source inconsistency: this redirect styles
            # neither "see" nor its target specially, so the usual 1Text
            # signal is absent. Recovered from raw text, flagged
            # SOURCE_ERROR since the source's own typography is what is
            # inconsistent here, not this extraction.
            tm = re.search(r"[Ss]ee\s+(.+?)\.?\s*$", text)
            if tm:
                xrefs = [CrossReference(
                    target=Sourced(value=tm.group(1).strip(), status="SOURCE_ERROR", decided_by=_DECIDED_BY),
                    type="redirect")]
                prov.decisions.append(Decision(
                    field_path="cross_references[0]", status="SOURCE_ERROR",
                    decided_by=_DECIDED_BY,
                    rationale="target carries none of the usual 1Text/see-italic "
                             "styling; recovered from raw text only"))
        if not xrefs:
            raise RuleBasedReaderError(
                f"entry_id={entry_id!r}: pointer-shaped but no cross-reference "
                "target is recoverable at all")
        return Entry(
            entry_id=entry_id, entry_type=entry_type,
            headword=Sourced(value=head, source_span=_span(para_idx), decided_by=_DECIDED_BY),
            homonym_index=homonym_index,
            cross_references=xrefs,
            provenance=prov,
        )

    headword, ipa_list, variants, offset, embedded_region = _extract_head(text, runs, context)
    # The homonym digit is a separate styled run with no space before the
    # pronunciation slash; strips only the exact digit homonym_index
    # already asserts, never a bare trailing-digit guess.
    if homonym_index is not None and headword.endswith(str(homonym_index)):
        headword = headword[:-len(str(homonym_index))].strip()
    labels, offset, pos_extensions = _extract_labels(runs, offset, context)
    rest_text = text[offset:]
    etymology, etym_extensions = _extract_etymology(runs, text, context)

    senses, per_sense_pos, per_sense_extensions, sense_variants = (
        _extract_senses(rest_text, runs, context, offset))
    if not senses:
        senses, per_sense_pos, per_sense_extensions, sense_variants = (
            _extract_senses(text, runs, context, 0))

    variant_forms = [
        VariantForm(form=Sourced(value=v, decided_by=_DECIDED_BY))
        for v in variants
    ] + sense_variants

    entry = Entry(
        entry_id=entry_id, entry_type=entry_type,
        headword=Sourced(value=headword, source_span=_span(para_idx), decided_by=_DECIDED_BY),
        homonym_index=homonym_index,
        # Entry-level pos ONLY when the source prints it once before sense
        # 1 -- when it prints POS again after each numbered sense instead,
        # sense extraction has already put it on the relevant sense, and
        # entry.pos stays empty rather than asserting a distribution the
        # source does not print.
        pos=[] if per_sense_pos else list(labels),
        variant_forms=variant_forms,
        pronunciations=[Pronunciation(ipa=Sourced(value=p, decided_by=_DECIDED_BY)) for p in ipa_list],
        etymology=etymology,
        senses=senses,
        cross_references=_extract_cross_refs(runs, is_pointer_entry=False),
        provenance=prov,
    )
    if embedded_region:
        # A region paren that landed inside the head span, before the
        # offset region extraction scans from -- rescued explicitly rather
        # than lost, same one-per-entry attachment as embedded labels.
        rescued = _extract_regions(embedded_region, context)
        for s in entry.senses:
            s.regions = rescued + s.regions

    all_pos_extensions = pos_extensions + per_sense_extensions
    if all_pos_extensions:
        entry.provenance.decisions.append(Decision(
            field_path="senses[].labels" if per_sense_pos else "pos",
            status="OBSERVED_EXTENSION", decided_by=_DECIDED_BY,
            rationale=f"grammatical-looking term(s) not in the closed "
                     f"vocabulary: {all_pos_extensions}",
        ))
    if etym_extensions:
        entry.provenance.decisions.append(Decision(
            field_path="etymology[].language_code", status="OBSERVED_EXTENSION",
            decided_by=_DECIDED_BY,
            rationale=f"language name(s) not in the closed vocabulary: {etym_extensions}",
        ))
    return entry
