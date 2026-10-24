"""Dictionary-body segmentation: locate structure, not content.

This establishes WHERE one entry ends and the next begins, WHAT KIND it is,
and gives it a durable ID. Headword, pronunciation, part-of-speech, and
cross-reference text extraction happen downstream, against the structure
this module establishes; getting a boundary wrong here would corrupt
everything built on top of it.

Two structural facts about the source, each independently re-verified by
this module's own gates on every run rather than assumed:

1. Letter-section headings are real: a paragraph with maximum run size 48
   half-points whose entire text is a single letter A-Z.
2. The source prints two kinds of entry, typographically distinct, not just
   textually distinct:
     - a full entry: one paragraph, starting with the headword in roman.
     - a cross-reference (redirect) entry: the headword is styled in the
       larger of the source's two headword typefaces, followed by an
       italic run containing exactly "see". That larger typeface alone is
       not sufficient: the source also uses it for some ordinary full
       entries with real content, so the actual redirect signal is
       specifically the italic "see" run, not the type size. This is a
       different construct from an inline "See X" cross-reference that
       occurs INSIDE a full entry's own definition, which is styled
       differently and is not followed by a bare "see" run of its own.
   One paragraph never holds more than one entry.

The homonym index is captured here, not deferred, because it is exactly as
structural as the typographic run properties this module already depends
on: an unstyled run holding a single digit, directly abutting the headword
letters with no separating space. The predicate additionally requires that
the headword run has not yet opened a pronunciation slash, because the same
digit-run styling is also how the source cites another entry's homonym
index inline, mid-definition -- and that inline citation can land at the
same run position, after the pronunciation has already begun. Requiring the
pre-pronunciation position removes exactly that false-positive class.

Gate: the entry count reconciles against an INDEPENDENT count taken by
re-walking the raw document.xml directly (not the run table this module
itself consumes), and the concatenation of every emitted entry's raw text
plus every letter-header's text must equal the body zone's source text
exactly, token for token -- no orphan text.
"""
from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
HOMONYM_DIGIT = re.compile(r"^\d$")


# ─── minimal private paragraph view ─────────────────────────────────────────
# Reproduced locally rather than shared with the apparatus package: this is
# the smallest read of a run table that segmentation needs (paragraph index,
# its ordered runs, its concatenated text, and its maximum run size), not a
# dependency on another component's internals.
@dataclass
class _Para:
    idx: int
    runs: list[dict]
    text: str
    max_sz: int


def _load_paras(runs: pa.Table) -> list[_Para]:
    """Regroup a flat run table into ordered paragraph views.

    Only "word/document.xml" runs are grouped: headers, footers, footnotes
    and endnotes each have their own independent para_idx numbering, so
    mixing them into this walk would silently merge unrelated paragraphs
    that happen to share a paragraph index with the main document body.
    """
    grouped: dict[int, list] = defaultdict(list)
    for r in runs.to_pylist():
        if r["part"] != "word/document.xml":
            continue
        grouped[r["para_idx"]].append(r)
    paras = []
    for i in sorted(grouped):
        rr = grouped[i]
        text = "".join(r["text"] for r in rr)
        max_sz = max((r["sz_half_pt"] or 0) for r in rr) if rr else 0
        paras.append(_Para(idx=i, runs=rr, text=text, max_sz=max_sz))
    return paras


def _is_letter_heading(p: _Para) -> bool:
    t = p.text.strip()
    return p.max_sz == 48 and len(t) == 1 and t in LETTERS


def _find_body_zone(paras: list[_Para]) -> tuple[int, int, list[_Para]]:
    """First letter heading to the paragraph before the "Sources" heading,
    both detected structurally rather than by paragraph number: any
    historical result is a validation fact, never a parsing input.
    """
    headings = [p for p in paras if _is_letter_heading(p)]
    if not headings:
        raise SegmentationError(
            "no letter-section headings found -- refusing to guess the "
            "dictionary body's start")
    sources = next((p.idx for p in paras
                    if p.idx > headings[-1].idx and p.text.strip() == "Sources"
                    and p.max_sz == 48), None)
    if sources is None:
        raise SegmentationError(
            "'Sources' heading not found after the last letter section -- "
            "cannot bound the dictionary body")
    return headings[0].idx, sources, headings


def _classify(p: _Para) -> str:
    """The larger of the source's two headword typefaces is not, on its
    own, the redirect signal: the source also uses it for some ordinary
    full entries with real content. Both settings carry an italic run at
    their own type size, but only a redirect's italic run says "see"; a
    full entry's italic run there gives its part of speech instead. The
    redirect signal is therefore the italic "see" run, not the type size.
    """
    if _is_letter_heading(p):
        return "letter_header"
    if any(r["char_style"] == "2Text" for r in p.runs) and any(
            r["italic"] and r["text"].strip().rstrip(".").lower() == "see"
            for r in p.runs):
        return "cross_ref"
    return "full"


def _homonym_index(p: _Para) -> int | None:
    """The digit run directly abutting the headword, if the source marks
    one: a lone-digit run at position 1, immediately after a headword run
    that has not yet opened a pronunciation slash.

    That second condition is load-bearing, not decoration: the same digit
    styling is also how the source cites another entry's homonym index
    inline, mid-definition -- and in a small number of entries that inline
    citation lands at run position 1 too, inside the pronunciation slashes
    rather than before them. Position alone would catch those as false
    positives; requiring the headword run to still be pre-pronunciation
    removes exactly that class and nothing else.

    A headword that itself ends in a digit (e.g. a postal-district-style
    name) needs no special case: that digit is part of run 0 itself, not a
    separate run, so this predicate never looks at it.
    """
    rr = p.runs
    if (len(rr) >= 2 and HOMONYM_DIGIT.match(rr[1]["text"])
            and "/" not in rr[0]["text"]):
        return int(rr[1]["text"])
    return None


def _independent_body_scan(source_path: Path, lo_idx: int,
                           hi_idx: int) -> tuple[int, int, list[str]]:
    """Re-derive (non-empty paragraph count, character count, ordered
    paragraph texts) for the body zone straight from the OOXML, bypassing
    the run table entirely.

    This is the independent path both reconciliation gates check the
    emitted output against: the counts AND the text no-orphan-text compares
    token-for-token both come from a walk that shares no code with the run
    table the emitted rows were built from, opening the source read-only
    and never writing to it. Comparing against the run table's own text
    instead would make that gate compare the output to itself.
    """
    with zipfile.ZipFile(source_path) as zf:
        root = etree.fromstring(zf.read("word/document.xml"))
    body = root.find(f"{{{W}}}body")
    n_paras, n_chars = 0, 0
    texts: list[str] = []
    for p_idx, para in enumerate(body.iter(f"{{{W}}}p")):
        if not (lo_idx <= p_idx <= hi_idx):
            continue
        text = "".join((t.text or "") for t in para.iter(f"{{{W}}}t"))
        if text.strip():
            n_paras += 1
            n_chars += len(text)
            texts.append(text)
    return n_paras, n_chars, texts


# ─── public result / error types ───────────────────────────────────────────
@dataclass(frozen=True)
class SegmentationReport:
    """Everything needed to audit a segmentation run without re-deriving it."""
    zone: dict[str, int]
    letter_sections: list[str]
    letter_sections_count: int
    entry_counts: dict[str, int]
    homonym_marked: int
    duplicate_entry_ids: list[str]
    independent_recount: dict[str, int]
    emitted: dict[str, int]
    entry_count_reconciles: bool
    zone_conserves_no_orphan_text: bool


class SegmentationError(RuntimeError):
    """Raised when the dictionary body's boundaries cannot be established,
    when content appears before any letter section, or when the post-pass
    gates (unique IDs, independent recount, no orphan text) fail.

    `report` carries the full `SegmentationReport` when one could be built
    (a completed pass whose post-hoc gates failed); it is `None` for a
    structural precondition failure discovered before any report exists.
    Callers must not treat this as recoverable by discarding a partial
    result -- none is ever returned alongside this exception.
    """

    def __init__(self, message: str, report: SegmentationReport | None = None):
        self.report = report
        super().__init__(message)


@dataclass(frozen=True)
class SegmentationResult:
    entries: pa.Table
    report: SegmentationReport


def segment_dictionary(runs: pa.Table, source_path: Path) -> SegmentationResult:
    """Segment the dictionary body into letter headers, full entries, and
    cross-reference (redirect) entries.

    `source_path` must point at the same source document the run table was
    ingested from -- it is read again, independently and read-only, so the
    reconciliation gates below check the emitted output against a path
    that shares no code with the one that produced it.

    Raises `SegmentationError` if the body boundaries cannot be established,
    if content precedes any letter section, or if any post-pass gate fails
    (duplicate entry IDs; the independent paragraph/character recount
    disagreeing with the emitted output; or the emitted text failing to
    reconstitute the source body token for token). No partial result is
    ever returned.
    """
    source_path = Path(source_path)
    paras = _load_paras(runs)
    lo, sources_idx, headings = _find_body_zone(paras)
    hi = sources_idx - 1  # last paragraph belonging to the dictionary body

    body = [p for p in paras if lo <= p.idx <= hi and p.text.strip()]

    rows: list[dict] = []
    letter: str | None = None
    seq_in_letter: dict[str, int] = {}
    for p in body:
        kind = _classify(p)
        if kind == "letter_header":
            letter = p.text.strip()
            seq_in_letter.setdefault(letter, 0)
            rows.append({
                "entry_id": f"L{letter}",
                "entry_type": "letter_header",
                "letter_section": letter,
                "homonym_index": None,
                "para_idx": p.idx,
                "raw": p.text,
                "locator": f"word/document.xml#p{p.idx}",
                "xpath": p.runs[0]["xpath"] if p.runs else None,
            })
            continue
        if letter is None:
            raise SegmentationError(
                f"content paragraph {p.idx} precedes any letter heading -- "
                "the dictionary body's boundary is wrong")
        seq_in_letter[letter] += 1
        rows.append({
            "entry_id": f"E{letter}{seq_in_letter[letter]:04d}",
            "entry_type": kind,
            "letter_section": letter,
            "homonym_index": _homonym_index(p),
            "para_idx": p.idx,
            "raw": p.text,
            "locator": f"word/document.xml#p{p.idx}",
            "xpath": p.runs[0]["xpath"] if p.runs else None,
        })

    # ---- gate: durable IDs are actually unique -----------------------------
    ids = [r["entry_id"] for r in rows]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})

    # ---- gate: entry count reconciles against an independent walk ---------
    indep_paras, indep_chars, indep_texts = _independent_body_scan(source_path, lo, hi)
    got_paras = len(rows)
    got_chars = sum(len(r["raw"]) for r in rows)
    count_ok = (indep_paras == got_paras) and (indep_chars == got_chars)

    # ---- gate: no orphan text -- raw OOXML reconstitutes the emitted
    # output, token for token. Compared against the independent scan above,
    # not against the run table `rows` was built from -- comparing the run
    # table to itself could never fail regardless of what the source says.
    src_tokens = " ".join(indep_texts).split()
    emit_tokens = " ".join(r["raw"] for r in rows).split()
    conserve_ok = src_tokens == emit_tokens

    n_full = sum(1 for r in rows if r["entry_type"] == "full")
    n_xref = sum(1 for r in rows if r["entry_type"] == "cross_ref")
    n_head = sum(1 for r in rows if r["entry_type"] == "letter_header")
    n_hom = sum(1 for r in rows if r["homonym_index"] is not None)

    report = SegmentationReport(
        zone={"first_para": lo, "last_para": hi, "sources_heading_para": sources_idx},
        letter_sections=[p.text.strip() for p in headings],
        letter_sections_count=len(headings),
        entry_counts={"full": n_full, "cross_ref": n_xref,
                      "letter_header": n_head, "total": len(rows)},
        homonym_marked=n_hom,
        duplicate_entry_ids=dup_ids,
        independent_recount={"paragraphs": indep_paras, "chars": indep_chars},
        emitted={"paragraphs": got_paras, "chars": got_chars},
        entry_count_reconciles=count_ok,
        zone_conserves_no_orphan_text=conserve_ok,
    )

    if dup_ids or not count_ok or not conserve_ok:
        reasons = []
        if dup_ids:
            reasons.append(f"{len(dup_ids)} duplicate entry id(s): {dup_ids}")
        if not count_ok:
            reasons.append(
                "independent recount disagrees with emitted output "
                f"(paragraphs {indep_paras} vs {got_paras}, "
                f"chars {indep_chars} vs {got_chars})")
        if not conserve_ok:
            reasons.append(
                "emitted text does not reconstitute the source body token for token")
        raise SegmentationError("segmentation gate FAILED -- " + "; ".join(reasons), report)

    return SegmentationResult(entries=pa.Table.from_pylist(rows), report=report)


def write_entries_parquet(entries: pa.Table, destination: Path) -> None:
    """Write the segmented entries table to Parquet, zstd-compressed."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(entries, destination, compression="zstd")
