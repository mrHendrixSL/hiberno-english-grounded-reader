"""Dictionary apparatus: parse the source's own front and back matter into
authority tables that constrain extraction downstream.

The source declares its own conventions rather than leaving them to be
inferred: the front matter gives the abbreviation vocabulary, the IPA
inventory, and the citation convention; the back matter gives the literary
sources, the key to non-literary sources, and the bibliography.

Both front-matter zones are real Word tables, so they are parsed by cell
coordinates rather than by guessing at paragraph alternation:

    tbl0  54 rows x 2 cols   term | expansion
    tbl1  29 rows x 3 cols   Vowels | Diphthongs | Consonants  (row 0 = heads)

A column in tbl1 *is* the phonetic category, which is why table context is
carried on every run by the ingestion stage. Zone boundaries for the back
matter are detected from heading text and display size, and are recorded in
the report rather than hardcoded as paragraph offsets.

Produces eleven authority tables:
    pos_labels, register_labels, language_codes, reference_sigla, symbols,
    editorial_abbrevs, ipa_inventory, citation_convention, literary_sources,
    non_literary_sources, bibliography

Gate: zero unexplained residue and zero unclassified abbreviations. Anything
unresolved is reported with a terminal status, never coerced.

This module is deliberately specific to this source's own apparatus. The
closed vocabularies, zone headings, and bibliographic parsing rules below
encode this dictionary's declared conventions and observed structural
irregularities -- they are not generic English-text heuristics, and
extending them to a different source is out of scope for this module.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ABBREV_TBL = 0
IPA_TBL = 1

# The abbreviation table is a closed, 54-row vocabulary from a fixed source, so
# membership is declared explicitly rather than inferred from wording. Anything
# outside these sets is reported as unclassified -- it must never be guessed,
# because these closed vocabularies are the declared authority vocabulary that
# automatic settlement of ambiguous cells later relies on.
POS_TERMS = {
    "adj.", "adv.", "adv.phr.", "conj.", "excl.", "imp.", "int.", "n.",
    "n.phr.", "p.part.", "phr.", "pl.", "prep.", "pres.part.", "v.", "v.n.",
    "voc.",
}
REGISTER_TERMS = {"colloq.", "dial.", "dimin.", "fig.", "lit.", "pejor."}
EDITORIAL_TERMS = {"cf.", "s.v."}
REFERENCE_TERMS = {"EDD", "ODEE", "OED"}
LANGUAGE_TERMS = {
    "AN", "Du", "E", "E dial.", "EModE", "F", "G", "Gk", "HE", "Ir", "It",
    "L", "LL", "MDu", "ME", "MF", "MIr", "ModE", "OE", "OF", "OIr", "ON",
    "ONF", "SE",
}


# ─── paragraph model ───────────────────────────────────────────────────────
class Para:
    __slots__ = ("idx", "runs", "text", "max_sz", "tbl", "row", "cell")

    def __init__(self, idx, runs):
        self.idx = idx
        self.runs = runs
        self.text = "".join(r["text"] for r in runs)
        self.max_sz = max((r["sz_half_pt"] or 0) for r in runs) if runs else 0
        self.tbl = runs[0]["tbl_idx"] if runs else None
        self.row = runs[0]["row_idx"] if runs else None
        self.cell = runs[0]["cell_idx"] if runs else None

    def italic_text(self) -> str:
        return "".join(r["text"] for r in self.runs if r["italic"]).strip()

    def styled(self, style) -> str:
        return "".join(r["text"] for r in self.runs
                       if r["char_style"] == style).strip()


def load_paras(runs: pa.Table) -> list[Para]:
    """Regroup a flat run table into ordered paragraph objects.

    Only "word/document.xml" runs are grouped. Headers, footers, footnotes
    and endnotes each have their own independent para_idx numbering, so
    mixing them into this walk would silently merge unrelated paragraphs
    that happen to share a paragraph index with the main document body.
    """
    rows = [r for r in runs.to_pylist() if r["part"] == "word/document.xml"]
    grouped: dict[int, list] = defaultdict(list)
    for r in rows:
        grouped[r["para_idx"]].append(r)
    return [Para(i, grouped[i]) for i in sorted(grouped)]


def cells(paras, tbl_idx) -> dict[tuple[int, int], list[Para]]:
    """Group a table's paragraphs by (row, cell)."""
    out: dict[tuple[int, int], list[Para]] = defaultdict(list)
    for p in paras:
        if p.tbl == tbl_idx and p.row is not None:
            out[(p.row, p.cell)].append(p)
    return out


def cell_text(grid, row, col) -> str:
    return " ".join(" ".join(p.text.split())
                    for p in grid.get((row, col), [])).strip()


# ─── zone detection (back matter is not tabular) ───────────────────────────
def find_zones(paras: list[Para]) -> dict[str, tuple[int, int]]:
    """Locate the four front/back-matter zones from six required landmark
    headings. Each landmark's presence is validated explicitly, in the
    order it must appear, before any arithmetic is done on its position:
    a missing or unrecoverable landmark is a structural precondition
    failure, raised as `ApparatusError`, never coerced into a wrong zone
    boundary by letting `None` reach a `+ 1` or a comparison.
    """
    if not paras:
        raise ApparatusError(
            "no paragraphs to search -- cannot locate any front/back-matter landmark")

    def find(pred, min_idx=0):
        for p in paras:
            if p.idx >= min_idx and pred(p):
                return p.idx
        return None

    def head(name, min_idx=0):
        return find(lambda p: p.max_sz == 48
                    and p.text.strip().lower() == name.lower(), min_idx)

    abbrev = head("Abbreviations and Symbols")
    if abbrev is None:
        raise ApparatusError('required landmark "Abbreviations and Symbols" not found')
    pron = head("Pronunciation Scheme")
    if pron is None:
        raise ApparatusError('required landmark "Pronunciation Scheme" not found')
    sources = head("Sources", min_idx=pron or 0)
    if sources is None:
        raise ApparatusError('required landmark "Sources" not found')
    lit = find(lambda p: p.text.strip() == "LITERARY SOURCES", sources or 0)
    if lit is None:
        raise ApparatusError('required landmark "LITERARY SOURCES" not found')
    nonlit = find(lambda p: p.text.strip().startswith(
        "KEY TO PRINCIPAL NON-LITERARY"), lit or 0)
    if nonlit is None:
        raise ApparatusError('required landmark "KEY TO PRINCIPAL NON-LITERARY..." not found')
    biblio = head("Bibliography", min_idx=nonlit or 0)
    if biblio is None:
        raise ApparatusError('required landmark "Bibliography" not found')

    return {
        "abbreviations": (abbrev + 1, pron),
        "literary_sources": (lit + 1, nonlit),
        "non_literary_sources": (nonlit + 1, biblio),
        "bibliography": (biblio + 1, paras[-1].idx + 1),
    }


def zone(paras, bounds, name):
    lo, hi = bounds[name]
    return [p for p in paras if lo <= p.idx < hi and p.text.strip()]


# ─── parsers ───────────────────────────────────────────────────────────────
def parse_abbrev_table(paras) -> tuple[list[dict], list[dict]]:
    """tbl0: one abbreviation per row, term in col 0, expansion in col 1."""
    grid = cells(paras, ABBREV_TBL)
    n_rows = max((r for r, _ in grid), default=-1) + 1
    rows, residue = [], []
    for r in range(n_rows):
        term = cell_text(grid, r, 0)
        expansion = cell_text(grid, r, 1)
        if not term and not expansion:
            continue
        if not term or not expansion:
            residue.append({"row": r, "term": term, "expansion": expansion,
                            "reason": "half-empty row"})
            continue
        para = grid[(r, 0)][0]
        rows.append({
            "term": term,
            "expansion": expansion,
            "expansion_title": (grid[(r, 1)][0].italic_text() or None
                                if (r, 1) in grid else None),
            "tbl_row": r,
            "para_idx": para.idx,
            "locator": f"word/document.xml#tbl{ABBREV_TBL}/r{r}",
        })
    return rows, residue


def classify_abbrev(term: str) -> str:
    if term in POS_TERMS:
        return "pos"
    if term in REGISTER_TERMS:
        return "register"
    if term in EDITORIAL_TERMS:
        return "editorial"
    if term in REFERENCE_TERMS:
        return "reference"
    if term in LANGUAGE_TERMS:
        return "language"
    if not any(c.isalnum() for c in term):
        return "symbol"
    # Deliberately NOT a catch-all: an unanticipated term must fail the gate
    # rather than be filed silently into a vocabulary that extraction later
    # treats as authoritative.
    return "unclassified"


def parse_ipa_table(paras) -> tuple[list[dict], list[dict]]:
    """tbl1: row 0 holds the column headings; a column IS the category."""
    grid = cells(paras, IPA_TBL)
    n_rows = max((r for r, _ in grid), default=-1) + 1
    sections = {c: cell_text(grid, 0, c).lower() or None for c in (0, 1, 2)}

    rows, residue = [], []
    for r in range(1, n_rows):
        for c in (0, 1, 2):
            t = cell_text(grid, r, c)
            if not t:
                continue
            m = re.match(r"^/\s*([^/]+?)\s*/\s*(.*)$", t)
            status, note = "DECLARED", None
            if not m:
                # A malformed line in the printed scheme (e.g. a missing
                # opening delimiter) is a source defect, not a parse failure.
                m = re.match(r"^\s*([^/\s]{1,6})\s*/\s*(.*)$", t)
                status = "SOURCE_ERROR"
                note = "opening delimiter absent in the source"
            if not m:
                residue.append({"row": r, "cell": c,
                                "locator": f"word/document.xml#tbl{IPA_TBL}/r{r}/c{c}"})
                continue
            sym = m.group(1).strip()
            para = grid[(r, c)][0]
            rows.append({
                "symbol": sym,
                "exemplar": m.group(2).strip(),
                "section": sections.get(c),
                "grapheme": para.styled("4Text") or None,
                "codepoints": " ".join(f"U+{ord(ch):04X}" for ch in sym),
                "names": " | ".join(unicodedata.name(ch, "?") for ch in sym),
                "status": status,
                "note": note,
                "para_idx": para.idx,
                "locator": f"word/document.xml#tbl{IPA_TBL}/r{r}/c{c}",
            })
    return rows, residue


SOURCE_LINE = re.compile(r"^([A-Z][A-Za-z]{0,5})\s*(?:,\s*([^:]+?))?\s*:\s*(.+)$",
                         re.S)
NEXT_ENTRY = re.compile(
    r"(?<=\s)(?=[A-Z]{2,5}(?:,\s*[A-Z][A-Za-z' ]{2,20})?\s*:\s)")
PUBLISHED = re.compile(
    r"[‘“].+?[’”]|\(\d{4}\)|\d{4}\)|Dictionary|Glossary"
    r"|Dialect|word\s*-?\s*list|Words and Phrases|Old Days|Fair Fingall")


def parse_non_literary(paras) -> tuple[list[dict], list[dict]]:
    """`CODE, County: Name`. The county is part of the identifier."""
    rows, residue = [], []
    for p in paras:
        t = " ".join(p.text.split())
        if not t:
            continue
        starts = [m.start() for m in NEXT_ENTRY.finditer(t)]
        chunks = ([t] if not starts
                  else [t[a:b].strip() for a, b in
                        zip([0] + starts, starts + [len(t)]) if t[a:b].strip()])
        for chunk in chunks:
            m = SOURCE_LINE.match(chunk)
            if not m:
                residue.append({"para_idx": p.idx,
                                "locator": f"word/document.xml#p{p.idx}"})
                continue
            name = m.group(3).strip()
            rows.append({
                "code": m.group(1),
                "county": (m.group(2) or "").strip() or None,
                "name": name,
                "source_type": ("publication" if PUBLISHED.search(name)
                                else "person"),
                "para_idx": p.idx,
                "locator": f"word/document.xml#p{p.idx}",
            })
    return rows, residue


TERMINAL = re.compile(r"[.\)\]]\s*$")

# A paragraph may end on a bare forename initial, or on an edited-collection
# byline (e.g. "...in A." or "Surname and Forename Surname (eds.),"), which
# TERMINAL cannot tell from a real full stop. Treated as terminal, the
# rejoin flushes mid-entry and the continuation is read as a fresh entry
# whose author is really the container's editor. "n.d." is NOT caught here,
# and must not be: it genuinely ends an entry.
INITIAL_TAIL = re.compile(r"(?<![\w.])[A-Z]\.\s*$")
YEAR = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")

# A quoted article title may itself contain a closing-quote character,
# because the apostrophe and the close-quote are the same glyph in this
# typography. Anchoring on the first close-quote truncates the title, so
# require the close to be followed by a real delimiter.
QUOTED = re.compile(r"[‘“](.{4,}?)[’”](?=\s*[,.;:]|\s*$|\s+in\s)")

# An entry may close with ")." or "]." rather than a bare "<digit>.", and a
# surname may begin with a lowercase nobiliary particle (e.g. "de", "van",
# "Mac"), so an uppercase-only lookahead would silently fuse those entries
# onto the previous one.
_NAME = r"(?:[A-ZÁÉÍÓÚÜÑÖÅ]|de\s|van\s|von\s|du\s|le\s|la\s|ó\s|Mac|Mc|O’)"
ENTRY_SPLIT = re.compile(
    r"(?:(?<=\d\.)|(?<=\d\)\.)|(?<=\d\]\.))\s+"
    r"(?=" + _NAME + r"[\w'’.\-]*(?:\s+[\w'’.\-]+)?,\s)")

# What the roman text BETWEEN two italic spans tells us about the second span.
CONTAINER_CUE = re.compile(r"\(eds?\.|\bin\s|\bed\.\s|\btrans\.\s")
CROSS_REF = re.compile(r"^(.{3,80}?),\s*see\s+(.+?)\.?\s*$")


def entry_like(text: str) -> bool:
    """A plausible bibliographic entry: terminated, dated, and substantial."""
    return bool(TERMINAL.search(text)) and bool(YEAR.search(text)) and len(text) > 40


def italic_spans(segs) -> tuple[list[str], list[str], list[int]]:
    """Split a segment list into contiguous italic spans and the roman text
    that separates them.

    Returns (spans, connectives, starts) where connectives[i] is the roman
    text immediately BEFORE spans[i] and starts[i] is its character offset in
    the concatenated segment text. Merging every italic run into one string
    instead fuses a title onto its container with no boundary between them,
    a genuine failure mode when two adjacent italic spans belong to
    different fields.
    """
    spans: list[str] = []
    connectives: list[str] = []
    starts: list[int] = []
    cur: list[str] = []
    roman: list[str] = []
    pos = 0
    cur_start = 0
    for text, is_ital in segs:
        if is_ital:
            if cur == []:
                connectives.append("".join(roman))
                roman = []
                cur_start = pos
            cur.append(text)
        else:
            if cur:
                spans.append("".join(cur))
                starts.append(cur_start)
                cur = []
            roman.append(text)
        pos += len(text)
    if cur:
        spans.append("".join(cur))
        starts.append(cur_start)
    # Whitespace is collapsed but punctuation is left alone: the comma that
    # ends a title span is the signal that a subtitle follows, and stripping it
    # here makes the joined title non-verbatim against `raw`.
    keep = [(i, " ".join(s.split())) for i, s in enumerate(spans)]
    keep = [(i, s) for i, s in keep if s.strip(" ,")]
    return ([s for _, s in keep],
            [connectives[i] if i < len(connectives) else "" for i, _ in keep],
            [starts[i] for i, _ in keep])


def parse_bibliographic(paras, kind) -> tuple[list[dict], list[dict], list[dict]]:
    """Literary sources and bibliography.

    Source realities the naive one-paragraph-per-entry reading gets wrong:

    1. Entries wrap across paragraphs. The DOCX flattened the hanging indent,
       so there is no structural signal; a paragraph that does not end in
       terminal punctuation is continued by the next one.
    2. One paragraph may hold SEVERAL complete entries.
    3. Article titles are quoted and roman while the *container* (journal or
       collection) is italic. Taking the italic run as the title therefore
       mislabels every article and pushes its real title into the author field.
    4. An entry may carry more than one italic span -- a container, a second
       work by the same author, or a continued subtitle.
    5. This is the ebook manifestation, so the back matter ends in publisher
       advertising which is not bibliography at all.

    Depends on the ingestion stage resolving `italic` by the ECMA-376 toggle
    rule; under nearest-wins these zones come back uniformly italic and every
    boundary below collapses.

    Returns (rows, residue, excluded_furniture).
    """
    # ---- 5. trim trailing non-bibliographic furniture, explicitly ----------
    end = len(paras)
    while end > 0 and not entry_like(" ".join(paras[end - 1].text.split())):
        end -= 1
    excluded = [{"para_idx": p.idx, "locator": f"word/document.xml#p{p.idx}",
                 "reason": "trailing non-bibliographic matter (ebook furniture)"}
                for p in paras[end:] if p.text.strip()]
    paras = paras[:end]

    # ---- 1. rejoin wrapped entries ----------------------------------------
    # Segments are kept as (text, is_italic) so the author/title boundary can
    # be found by run position rather than by locating the italic string inside
    # the text -- that lookup is fragile under whitespace and punctuation
    # normalisation and silently dropped the author on a non-trivial number of
    # entries.
    joined: list[dict] = []
    buf: dict | None = None
    for p in paras:
        if not p.text.strip():
            continue
        segs = [(r["text"], bool(r["italic"])) for r in p.runs]
        if buf is None:
            buf = {"segs": list(segs), "para_idx": p.idx, "wrapped": False}
        else:
            buf["segs"].append((" ", False))
            buf["segs"].extend(segs)
            buf["wrapped"] = True
        so_far = "".join(s for s, _ in buf["segs"]).rstrip()
        if TERMINAL.search(so_far) and not INITIAL_TAIL.search(so_far):
            joined.append(buf)
            buf = None
    if buf is not None:
        joined.append(buf)

    # ---- 2. split paragraphs that hold more than one entry ----------------
    expanded: list[dict] = []
    for e in joined:
        full = "".join(s for s, _ in e["segs"])
        # Cut offsets come from the match objects, not from summing the lengths
        # of re.split's parts: the pattern CONSUMES the whitespace between
        # entries, so a length sum drifts left by one separator per extra cut
        # and pushes the previous entry's final "." onto the next one.
        offsets = [m.end() for m in ENTRY_SPLIT.finditer(full)]
        if not offsets:
            expanded.append(e)
            continue
        # Re-slice the segment list at those offsets so the italic flags stay
        # aligned with their text.
        cut, seg_pos, current = 0, 0, []
        for s, is_i in e["segs"]:
            start = seg_pos
            for ch_i, ch in enumerate(s):
                if cut < len(offsets) and start + ch_i >= offsets[cut]:
                    expanded.append({"segs": current, "para_idx": e["para_idx"],
                                     "wrapped": e["wrapped"], "split": True})
                    current = []
                    cut += 1
                current.append((ch, is_i))
            seg_pos += len(s)
        if current:
            expanded.append({"segs": current, "para_idx": e["para_idx"],
                             "wrapped": e["wrapped"], "split": True})
    joined = expanded

    # ---- 3/4. split author / title / container ----------------------------
    rows, residue = [], []
    for e in joined:
        text = " ".join("".join(s for s, _ in e["segs"]).split())
        raw_full = "".join(s for s, _ in e["segs"])

        spans, connectives, starts = italic_spans(e["segs"])

        # character offset at which the first italic run begins
        ital_start, pos = None, 0
        for s, is_i in e["segs"]:
            if is_i and s.strip() and ital_start is None:
                ital_start = pos
            pos += len(s)

        # A line matching "<work>, see <target>." is a pointer, not a work
        # in its own right: it has no title, author, imprint or date of
        # its own.
        xref = CROSS_REF.match(text)
        if xref and not YEAR.search(text):
            rows.append({
                "kind": kind, "record_type": "cross_reference",
                "author": None, "author_method": None,
                "title": xref.group(1).strip(), "title_method": "structural",
                "title_status": "SETTLED",
                "see_ref": xref.group(2).strip(),
                "container": None, "additional_titles": None,
                "year_status": None, "year_source_para": None,
                "year_note": None,
                "split_from_shared_paragraph": e.get("split", False),
                "year": None, "wrapped": e["wrapped"], "raw": text,
                "para_idx": e["para_idx"],
                "locator": f"word/document.xml#p{e['para_idx']}",
            })
            continue

        # A quoted span at the very start is a pseudonymous author, not a
        # title; the real title is then the next quoted span.
        quoted_all = list(QUOTED.finditer(raw_full))
        pseudonym = None
        if quoted_all and quoted_all[0].start() <= 1:
            pseudonym = " ".join(quoted_all[0].group(1).split()).strip()
            quoted_all = quoted_all[1:]
        quoted = quoted_all[0] if quoted_all else None
        q_start = quoted.start() if quoted else None
        candidates = [c for c in (q_start, ital_start) if c is not None]
        cut_raw = min(candidates) if candidates else None

        title_status = "SETTLED"
        additional: list[str] = []
        container = None
        title_method = "structural"

        if quoted and (ital_start is None or q_start < ital_start):
            # Article: quoted title ROMAN, container italic. The inequality is
            # strict on purpose. A monograph title may itself OPEN with a
            # quotation, and there the quote lies inside the italic span
            # rather than preceding it, so the two offsets coincide. Reading
            # that as an article truncates the title and promotes its tail
            # to container.
            title = " ".join(quoted.group(1).split()).strip()
            # Only italic that begins AFTER the quoted title can be the
            # container. An italic span inside the quotes is part of the title
            # itself -- an Irish etymon, or "c." for circa.
            outside = [s for s, st in zip(spans, starts) if st >= quoted.end()]
            if outside:
                container = "; ".join(s.strip(" ,") for s in outside)
        elif spans:
            title = spans[0]
            rest = spans[1:]
            for i, extra in enumerate(rest, start=1):
                link = connectives[i] if i < len(connectives) else ""
                if CONTAINER_CUE.search(link):
                    ex = extra.strip(" ,")
                    container = ex if container is None else f"{container}; {ex}"
                elif title.rstrip().endswith(",") or link.strip() == "" or (
                        not link.strip().startswith(("(", "["))
                        and not YEAR.search(link) and "and" not in link.split()):
                    # A continued subtitle: the roman text between the spans
                    # is part of the title, not a link to a second, separate
                    # work.
                    title = f"{title}{link}{extra}"
                    title = " ".join(title.split())
                else:
                    additional.append(extra.strip(" ,"))
            title = title.strip(" ,")
            if additional:
                title_status = "AMBIGUOUS"
        else:
            # Derive: text after the author, up to the imprint ("Place:") or
            # a trailing year. Flagged so downstream can tell it apart.
            rest = text
            m_auth = re.match(r"^\s*([^,]{2,40},\s*[^,]{1,40}?"
                              r"(?:\s*\([^)]{0,80}\))?)\s*,\s*", rest)
            if m_auth:
                rest = rest[m_auth.end():]
            m_imp = re.search(r",\s*[A-Z][\w'\. ]{2,30}:\s|,\s*\d{4}", rest)
            title = (rest[:m_imp.start()] if m_imp else rest).strip(" ,.")
            title_method = "derived"

        # map the raw offset onto the whitespace-collapsed text
        author = (" ".join(raw_full[:cut_raw].split()).strip(" ,")
                  if cut_raw else None)
        author_method = "structural" if author else None
        if not author and pseudonym:
            author, author_method = pseudonym, "pseudonym"
        if not author:
            # The source is inconsistent: some entries italicise the author as
            # well as the title, so there is no roman prefix to cut at. Fall
            # back to the bibliographic convention "Surname, Forename," and
            # mark the value as heuristic so downstream can tell them apart.
            # A parenthetical after the name may itself contain commas (e.g.
            # a co-editor credit), so it must be consumed as part of the
            # name, not split on its internal comma.
            m = re.match(r"^\s*([^,(]{2,40},\s*[^,(]{1,40}?"
                         r"(?:\s*\([^)]{0,80}\))?)\s*,", text)
            if m:
                author, author_method = m.group(1).strip(), "heuristic"
        # Post-check: author and title must be disjoint. Overlap means the
        # split was wrong -- either the work is anonymous and its title was
        # mistaken for "Surname, Forename", or the author prefix was left
        # inside the title.
        if author and title:
            if title.strip() == author.strip():
                author, author_method = None, None      # anonymous title-first
            elif title.startswith(author):
                q_any = QUOTED.search(raw_full)
                if q_any:
                    title = " ".join(q_any.group(1).split()).strip()
                else:
                    title = title[len(author):].strip(" ,.")
                title_method = "derived"

        years = YEAR.findall(text)
        rows.append({
            "kind": kind,
            "record_type": "work",
            "author": author or None,
            "author_method": author_method,
            "title": title or None,
            "title_method": title_method,
            "title_status": title_status,
            "see_ref": None,
            "container": container,
            "additional_titles": "; ".join(additional) if additional else None,
            "split_from_shared_paragraph": e.get("split", False),
            "year": int(years[-1]) if years else None,
            "year_status": None, "year_source_para": None, "year_note": None,
            "wrapped": e["wrapped"],
            "raw": text,
            "para_idx": e["para_idx"],
            "locator": f"word/document.xml#p{e['para_idx']}",
        })
        if not title:
            residue.append({"para_idx": e["para_idx"],
                            "locator": f"word/document.xml#p{e['para_idx']}",
                            "reason": "no title resolved"})
    return rows, residue, excluded


def _citation_convention_template() -> list[dict]:
    """A fresh copy of the three machine-readable citation-convention
    records, before the source paragraph that states them is attached.

    Fields are machine-readable only: `marker`/`chars`/`citation_class`
    identify the convention, `source_para_idx`/`source_locator` point at
    where the source states it. The source's own prose statement of each
    convention is not carried here -- nothing downstream consumes it, and
    a public diagnostic table should not carry unused source prose.
    """
    return [
        {"marker": "double_quote", "chars": "“ ”", "citation_class": "literary"},
        {"marker": "single_quote", "chars": "‘ ’",
         "citation_class": "oral_or_correspondent"},
        {"marker": "small_capitals", "chars": "",
         "citation_class": "cross_reference"},
    ]


BAD_BRACKET = re.compile(r"\[(\d{5,}|\d{1,3})\]")
NO_DATE = re.compile(r"(?<![\w.])n\.\s?d\.")


def _tkey(s: str | None) -> str:
    """Normalise a title for matching: case, punctuation and spacing only."""
    if not s:
        return ""
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def classify_years(rows: list[dict]) -> None:
    """Decide, for every entry, WHY it has no year -- in place.

    An absent year is not one fact but five, and collapsing them loses the
    distinction between a date the source declares unknowable, a date it
    misprints, and a date it simply files elsewhere:

      SETTLED       a year was read from the entry.
      OUT_OF_SCHEMA the record is a cross-reference; a pointer has no date.
      DECLARED      the source itself says the date is unknown ("n.d.").
      SOURCE_ERROR  the source prints something that cannot be a year -- e.g.
                    "[19001902]", an en-dash lost from "[1900-1902]".
      DEFERRED      a chapter in a collection: the date lives at the
                    collection's OWN entry in this same bibliography, and
                    `year_source_para` records which one.
      AMBIGUOUS     no year, and no entry in this bibliography supplies one.

    DEFERRED is resolved structurally, never by inference: the container title
    must match a dated entry's title -- exactly, or as a prefix ending at a
    subtitle boundary -- AND a surname from that entry's author must appear in
    this entry's raw text. Both signals are required because a bare title
    prefix is not distinctive (a short prefix can reach several unrelated
    entries) and a bare surname is not either.
    """
    dated: dict[str, list[dict]] = {}
    for r in rows:
        if r["title"] and r["year"]:
            dated.setdefault(_tkey(r["title"]), []).append(r)

    def resolve(row) -> dict | None:
        ck = _tkey(row["container"])
        if not ck:
            return None
        cands = list(dated.get(ck, []))
        if not cands:
            # The cited container title is usually the short form, so match on
            # a word-boundary prefix. That alone is not distinctive -- the
            # same prefix can reach several unrelated entries -- which is why
            # the surname test below is not optional.
            for k, v in dated.items():
                if k.startswith(ck + " "):
                    cands.extend(v)
        hits = []
        for c in cands:
            surnames = set(re.findall(r"[^\W\d_]{3,}", c["author"] or ""))
            if any(s in row["raw"] for s in surnames):
                hits.append(c)
        # Two survivors mean the pair of signals did not single an entry out;
        # guessing between them would be exactly the inference this avoids.
        uniq = {h["para_idx"] for h in hits}
        return hits[0] if len(uniq) == 1 else None

    for r in rows:
        if r["year"]:
            r["year_status"] = "SETTLED"
            continue
        if r["record_type"] == "cross_reference":
            r["year_status"] = "OUT_OF_SCHEMA"
            r["year_note"] = "a cross-reference is a pointer, not a work"
            continue
        if NO_DATE.search(r["raw"]):
            r["year_status"] = "DECLARED"
            r["year_note"] = "the source prints n.d."
            continue
        bad = BAD_BRACKET.search(r["raw"])
        if bad:
            r["year_status"] = "SOURCE_ERROR"
            r["year_note"] = (f"bracketed date {bad.group(0)} is not a year; "
                              "a range separator appears to be lost")
            continue
        src_row = resolve(r)
        if src_row is not None:
            r["year_status"] = "DEFERRED"
            r["year_source_para"] = src_row["para_idx"]
            r["year_note"] = (f"dated at the container's own entry "
                              f"(p{src_row['para_idx']}, {src_row['year']})")
            continue
        r["year_status"] = "AMBIGUOUS"
        r["year_note"] = (
            "container cited but no dated entry in this bibliography matches "
            f"it ({r['container']!r})" if r["container"] else
            "no year in the entry and no container to defer to")


# ─── public result / error types ───────────────────────────────────────────
@dataclass(frozen=True)
class ApparatusReport:
    """Everything needed to audit an extraction run without re-deriving it."""
    zones_detected: dict[str, dict[str, int | None]]
    tables_used: dict[str, int]
    table_rows: dict[str, int]
    abbreviations_total: int
    abbreviations_by_kind: dict[str, int]
    unclassified_abbreviations: list[str]
    ipa_sections: dict[str, int]
    ipa_status: dict[str, int]
    non_literary: dict[str, int]
    bibliographic_titles_resolved: dict[str, int]
    bibliographic_methods: dict[str, dict[str, int]]
    year_status: dict[str, dict[str, int]]
    year_unsettled_detail: list[dict]
    excluded_furniture: dict[str, list[dict]]
    wrapped_entries_rejoined: dict[str, int]
    unparsed_residue: dict[str, int]
    residue_total: int
    residue_detail: dict[str, list[dict]]


class ApparatusError(RuntimeError):
    """Raised when the source's front/back-matter structure cannot be
    established, or when the post-parse gate fails: unexplained residue, or
    an abbreviation the closed vocabularies do not account for.

    `report` carries the full `ApparatusReport` when one could be built (a
    completed parse whose gate failed); it is `None` for a structural
    precondition failure discovered before any report exists (e.g. a
    missing required landmark heading). Callers must not treat this as
    recoverable by discarding the offending rows; the extraction is
    incomplete until the source apparatus itself is understood, and no
    partial result is ever returned alongside this exception.
    """

    def __init__(self, message: str, report: ApparatusReport | None = None):
        self.report = report
        super().__init__(message)


@dataclass(frozen=True)
class ApparatusResult:
    tables: dict[str, pa.Table]
    report: ApparatusReport


def extract_authorities(runs: pa.Table) -> ApparatusResult:
    """Parse a run table's front/back matter into the eleven authority
    tables.

    Raises `ApparatusError` if any source content is left as unexplained
    residue, or if any abbreviation falls outside the closed vocabularies --
    a caller must not receive a partially-classified result.
    """
    paras = load_paras(runs)
    bounds = find_zones(paras)

    abbrev_rows, abbrev_res = parse_abbrev_table(paras)
    for r in abbrev_rows:
        r["kind"] = classify_abbrev(r["term"])
    ipa_rows, ipa_res = parse_ipa_table(paras)

    # The convention is stated in prose after the abbreviations table.
    conv = next((p for p in zone(paras, bounds, "abbreviations")
                 if "quotation marks" in p.text
                 and "SMALL CAPITAL" in p.text.upper()), None)
    citation_convention = _citation_convention_template()
    for rule in citation_convention:
        rule["source_para_idx"] = conv.idx if conv else None
        rule["source_locator"] = (f"word/document.xml#p{conv.idx}"
                                  if conv else None)

    nl_rows, nl_res = parse_non_literary(zone(paras, bounds, "non_literary_sources"))
    lit_rows, lit_res, lit_excl = parse_bibliographic(
        zone(paras, bounds, "literary_sources"), "literary")
    bib_rows, bib_res, bib_excl = parse_bibliographic(
        zone(paras, bounds, "bibliography"), "bibliography")

    # Resolved per table: a literary source's container is a literary source.
    classify_years(lit_rows)
    classify_years(bib_rows)

    by_kind = defaultdict(list)
    for r in abbrev_rows:
        by_kind[r["kind"]].append(r)

    tables = {
        "pos_labels": pa.Table.from_pylist(by_kind["pos"]),
        "register_labels": pa.Table.from_pylist(by_kind["register"]),
        "language_codes": pa.Table.from_pylist(by_kind["language"]),
        "reference_sigla": pa.Table.from_pylist(by_kind["reference"]),
        "symbols": pa.Table.from_pylist(by_kind["symbol"]),
        "editorial_abbrevs": pa.Table.from_pylist(by_kind["editorial"]),
        "ipa_inventory": pa.Table.from_pylist(ipa_rows),
        "citation_convention": pa.Table.from_pylist(citation_convention),
        "literary_sources": pa.Table.from_pylist(lit_rows),
        "non_literary_sources": pa.Table.from_pylist(nl_rows),
        "bibliography": pa.Table.from_pylist(bib_rows),
    }
    table_rows = {k: v.num_rows for k, v in tables.items()}

    residue = {"abbreviations": abbrev_res, "ipa": ipa_res,
               "non_literary_sources": nl_res,
               "literary_sources": lit_res, "bibliography": bib_res}
    total_residue = sum(len(v) for v in residue.values())

    ambiguous = defaultdict(list)
    for r in nl_rows:
        ambiguous[r["code"]].append(r["county"])
    amb = {c: v for c, v in ambiguous.items() if len(v) > 1}

    unclassified = by_kind.get("unclassified", [])

    report = ApparatusReport(
        zones_detected={k: {"first_para": v[0], "last_para": v[1] - 1}
                        for k, v in bounds.items()},
        tables_used={"abbreviations": ABBREV_TBL, "ipa": IPA_TBL},
        table_rows=table_rows,
        abbreviations_total=len(abbrev_rows),
        abbreviations_by_kind={k: len(v) for k, v in by_kind.items()},
        unclassified_abbreviations=[r["term"] for r in unclassified],
        # Sorted, not built over a `set`: set iteration order depends on
        # PYTHONHASHSEED, which is randomised per process, so building this
        # over a `set` makes the report's key order change on every re-run
        # even when the underlying data is identical.
        ipa_sections={s: sum(1 for r in ipa_rows if r["section"] == s)
                      for s in sorted({r["section"] for r in ipa_rows})},
        ipa_status={s: sum(1 for r in ipa_rows if r["status"] == s)
                    for s in sorted({r["status"] for r in ipa_rows})},
        non_literary={
            "total": len(nl_rows),
            "persons": sum(1 for r in nl_rows if r["source_type"] == "person"),
            "publications": sum(1 for r in nl_rows
                                if r["source_type"] == "publication"),
            "codes_ambiguous_without_county": len(amb),
        },
        bibliographic_titles_resolved={
            "literary": sum(1 for r in lit_rows if r["title"]),
            "bibliography": sum(1 for r in bib_rows if r["title"]),
        },
        # Dispositions, not just counts: a table is only trustworthy if the
        # shape of what it could NOT settle is visible on its face.
        bibliographic_methods={
            k: dict(Counter(r[f] for r in rows))
            for k, rows in (("literary", lit_rows), ("bibliography", bib_rows))
            for f in ("author_method",)
        } | {
            f"{k}_title_status": dict(Counter(r["title_status"] for r in rows))
            for k, rows in (("literary", lit_rows), ("bibliography", bib_rows))
        },
        year_status={
            k: dict(Counter(r["year_status"] for r in rows))
            for k, rows in (("literary", lit_rows), ("bibliography", bib_rows))
        },
        year_unsettled_detail=[
            {"para_idx": r["para_idx"], "locator": r["locator"],
             "status": r["year_status"],
             "source_para": r["year_source_para"], "note": r["year_note"]}
            for r in lit_rows + bib_rows if r["year_status"] != "SETTLED"
        ],
        excluded_furniture={"literary": lit_excl, "bibliography": bib_excl},
        wrapped_entries_rejoined={
            "literary": sum(1 for r in lit_rows if r["wrapped"]),
            "bibliography": sum(1 for r in bib_rows if r["wrapped"]),
        },
        unparsed_residue={k: len(v) for k, v in residue.items()},
        residue_total=total_residue,
        residue_detail=residue,
    )

    if total_residue or unclassified:
        reasons = []
        if total_residue:
            reasons.append(f"{total_residue} unexplained residue row(s)")
        if unclassified:
            terms = [r["term"] for r in unclassified]
            reasons.append(f"{len(terms)} unclassified abbreviation(s): {terms}")
        raise ApparatusError("apparatus gate FAILED -- " + "; ".join(reasons), report)

    return ApparatusResult(tables=tables, report=report)


def write_authority_tables(tables: dict[str, pa.Table], destination: Path) -> None:
    """Write the eleven authority tables to Parquet, zstd-compressed."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        pq.write_table(table, destination / f"{name}.parquet", compression="zstd")
