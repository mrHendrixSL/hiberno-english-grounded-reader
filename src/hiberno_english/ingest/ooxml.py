"""Raw OOXML ingestion: DOCX -> typed run stream.

Reads a .docx file as a raw OPC package (zipfile + lxml), NOT via a
higher-level DOCX library, because those abstract away exactly the style
inheritance and XML constructs this module depends on. It emits one row per
text-bearing run, in document order, carrying the exact text plus the
*effective* character formatting resolved through the inheritance chain
docDefaults -> paragraph style -> character style -> direct run properties.

Recovering the effective formatting (not just the raw text) matters whenever
a document's typography is functional rather than decorative -- e.g. a
source that consistently italicises one category of information and uses a
reduced-size run style for another. This module resolves that formatting
correctly; deciding what any particular style means downstream is the
caller's concern, not this module's.

Conservation gate: every <w:t> node in every text-bearing part must appear
in the output, with identical character counts. No normalisation of any
kind -- code points and order are preserved exactly.
"""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


# Character properties carried forward. Toggles are tri-state (None = inherit).
TOGGLES = ("i", "b", "smallCaps", "caps", "strike")
VALUES = ("vertAlign", "sz", "rStyle")


def parse_rpr(rpr) -> dict:
    """Extract the character properties tracked here from a <w:rPr> element."""
    out: dict = {}
    if rpr is None:
        return out
    for name in TOGGLES:
        el = rpr.find(q(name))
        if el is not None:
            val = el.get(q("val"))
            # Absent @val means "on"; explicit 0/false means "off".
            out[name] = val not in ("0", "false")
    for name in VALUES:
        el = rpr.find(q(name))
        if el is not None:
            out[name] = el.get(q("val"))
    rfonts = rpr.find(q("rFonts"))
    if rfonts is not None:
        out["font"] = rfonts.get(q("ascii")) or rfonts.get(q("hAnsi"))
    return out


def build_style_index(styles_root) -> tuple[dict, dict]:
    """Map styleId -> {type, basedOn, rpr}, plus the docDefaults run props."""
    index = {}
    for style in styles_root.findall(q("style")):
        sid = style.get(q("styleId"))
        based = style.find(q("basedOn"))
        index[sid] = {
            "type": style.get(q("type")),
            "basedOn": based.get(q("val")) if based is not None else None,
            "rpr": parse_rpr(style.find(q("rPr"))),
            "name": (style.find(q("name")).get(q("val"))
                     if style.find(q("name")) is not None else None),
        }
    defaults = {}
    dd = styles_root.find(q("docDefaults"))
    if dd is not None:
        rpr_default = dd.find(q("rPrDefault"))
        if rpr_default is not None:
            defaults = parse_rpr(rpr_default.find(q("rPr")))
    return index, defaults


def _apply(eff: dict, layer: dict) -> None:
    """Fold one style layer into the accumulator.

    Toggle properties XOR; everything else is nearest-wins.
    """
    for name, val in layer.items():
        if name in TOGGLES:
            eff[name] = bool(eff.get(name, False)) ^ bool(val)
        else:
            eff[name] = val


def resolve_style(style_id: str | None, index: dict, _seen=None) -> dict:
    """Merge a style's properties with its basedOn ancestors (root first)."""
    if not style_id or style_id not in index:
        return {}
    _seen = _seen or set()
    if style_id in _seen:                       # cycle guard
        return {}
    _seen.add(style_id)
    entry = index[style_id]
    merged = resolve_style(entry["basedOn"], index, _seen)
    _apply(merged, entry["rpr"])
    return merged


def effective_props(direct: dict, para_style: str | None, index: dict,
                     defaults: dict) -> dict:
    """Effective character properties, per ECMA-376 Part 1 §17.7.3.

    `w:i`, `w:b`, `w:caps`, `w:smallCaps` and `w:strike` are *toggle*
    properties: where several levels of the style hierarchy each specify
    one, the effective value is their XOR, not the nearest one. Direct run
    formatting is absolute and ends the chain.

    This is load-bearing, not pedantry: a source can use an enclosing
    paragraph style that is itself italic, with an explicit character style
    on some runs that is ALSO nominally "italic" -- under XOR resolution
    those two cancel, so the explicit style marks the ROMAN span and a bare
    run inherits the paragraph's italic. Reading this as nearest-wins
    instead would mark the whole zone italic and destroy that boundary.
    """
    eff = dict(defaults)
    _apply(eff, resolve_style(para_style, index))
    _apply(eff, resolve_style(direct.get("rStyle"), index))
    # Direct formatting overrides outright, toggles included.
    for name, val in direct.items():
        if name != "rStyle":
            eff[name] = val
    return eff


def text_parts(zf: zipfile.ZipFile) -> list[str]:
    """Every part that can hold body text, in a stable order."""
    candidates = [n for n in zf.namelist()
                  if n.startswith("word/") and n.endswith(".xml")]
    keep = []
    for n in candidates:
        base = n.rsplit("/", 1)[-1]
        if base == "document.xml" or base.startswith(
                ("footnotes", "endnotes", "header", "footer")):
            keep.append(n)
    return sorted(keep, key=lambda n: (n != "word/document.xml", n))


@dataclass(frozen=True)
class ConservationReport:
    """Whether every source text node was emitted, with every character.

    Identifies the source by name and content digest, never by filesystem
    path: a public diagnostic report must not leak the local directory
    layout it was produced under.
    """
    source_name: str
    source_sha256: str
    parts_read: list[str]
    expected_text_nodes: int
    emitted_rows: int
    expected_chars: int
    emitted_chars: int
    conservation_pass: bool
    per_part: dict[str, dict[str, int]]
    character_styles_defined: int


class ConservationError(RuntimeError):
    """Raised when the emitted run stream does not conserve the source.

    The caller must not receive a partial or silently-truncated result:
    a conservation failure is a defect in extraction, not a warning.
    """

    def __init__(self, report: ConservationReport):
        self.report = report
        super().__init__(
            "conservation FAILED: expected "
            f"{report.expected_text_nodes:,} text nodes / {report.expected_chars:,} "
            f"characters, emitted {report.emitted_rows:,} rows / "
            f"{report.emitted_chars:,} characters"
        )


@dataclass(frozen=True)
class IngestResult:
    runs: pa.Table
    conservation: ConservationReport


RUN_SCHEMA_FIELDS = (
    "run_uid", "part", "para_idx", "run_idx", "text", "n_chars",
    "space_preserve", "para_style", "tbl_idx", "row_idx", "cell_idx",
    "char_style", "italic", "bold", "small_caps", "all_caps", "vert_align",
    "sz_half_pt", "font", "locator", "xpath",
)


def ingest_docx(source_path: Path) -> IngestResult:
    """Ingest one .docx file into a typed, document-ordered run table.

    Raises `ConservationError` if the emitted text-node count or character
    count does not match what is present in the source XML -- callers must
    not treat a `ConservationError` as recoverable; it means the run stream
    is incomplete.
    """
    source_path = Path(source_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    with zipfile.ZipFile(source_path) as zf:
        styles_root = etree.fromstring(zf.read("word/styles.xml"))
        style_index, defaults = build_style_index(styles_root)
        parts = text_parts(zf)
        part_bytes = {p: zf.read(p) for p in parts}

    rows: list[dict] = []
    expected: dict[str, tuple[int, int]] = {}
    global_order = 0

    for part in parts:
        root = etree.fromstring(part_bytes[part])
        body = root.find(q("body")) if root.find(q("body")) is not None else root

        # Table context: a real <w:tbl> structure can carry categorical
        # meaning per column, so cell coordinates are recorded whenever a
        # paragraph sits inside one. Innermost table wins, since a nested
        # tbl is visited after its parent.
        #
        # Keyed by canonical XPath, NOT id(): lxml hands out fresh proxy
        # objects for the same underlying node, so identity is unreliable.
        tree = root.getroottree()
        cell_of: dict[str, tuple[int, int, int]] = {}
        for t_i, tbl in enumerate(root.iter(q("tbl"))):
            for r_i, tr in enumerate(tbl.findall(q("tr"))):
                for c_i, tc in enumerate(tr.findall(q("tc"))):
                    for para in tc.iter(q("p")):
                        cell_of[tree.getpath(para)] = (t_i, r_i, c_i)

        # Independent expectation, measured on the raw XML before we walk it.
        all_t = root.iter(q("t"))
        n_nodes = 0
        n_chars = 0
        for t in all_t:
            n_nodes += 1
            n_chars += len(t.text or "")
        expected[part] = (n_nodes, n_chars)

        for p_idx, para in enumerate(body.iter(q("p"))):
            ppr = para.find(q("pPr"))
            pstyle = None
            if ppr is not None:
                el = ppr.find(q("pStyle"))
                if el is not None:
                    pstyle = el.get(q("val"))
            para_path = tree.getpath(para)
            tbl_idx, row_idx, cell_idx = cell_of.get(para_path,
                                                     (None, None, None))

            r_idx = 0
            for run in para.iter(q("r")):
                direct = parse_rpr(run.find(q("rPr")))
                eff = effective_props(direct, pstyle, style_index, defaults)
                for t in run.findall(q("t")):
                    text = t.text or ""
                    rows.append({
                        "run_uid": global_order,
                        "part": part,
                        "para_idx": p_idx,
                        "run_idx": r_idx,
                        "text": text,
                        "n_chars": len(text),
                        "space_preserve": t.get(f"{{{XML}}}space") == "preserve",
                        "para_style": pstyle,
                        "tbl_idx": tbl_idx,
                        "row_idx": row_idx,
                        "cell_idx": cell_idx,
                        "char_style": direct.get("rStyle"),
                        "italic": eff.get("i"),
                        "bold": eff.get("b"),
                        "small_caps": eff.get("smallCaps"),
                        "all_caps": eff.get("caps"),
                        "vert_align": eff.get("vertAlign"),
                        "sz_half_pt": (int(eff["sz"]) if str(eff.get("sz", "")).isdigit()
                                       else None),
                        "font": eff.get("font"),
                        "locator": f"{part}#p{p_idx}/r{r_idx}",
                        "xpath": para_path,
                    })
                    global_order += 1
                r_idx += 1

    # ---- conservation gate -------------------------------------------------
    got_nodes = len(rows)
    got_chars = sum(r["n_chars"] for r in rows)
    exp_nodes = sum(v[0] for v in expected.values())
    exp_chars = sum(v[1] for v in expected.values())
    ok = (got_nodes == exp_nodes) and (got_chars == exp_chars)

    report = ConservationReport(
        source_name=source_path.name,
        source_sha256=source_sha256,
        parts_read=parts,
        expected_text_nodes=exp_nodes,
        emitted_rows=got_nodes,
        expected_chars=exp_chars,
        emitted_chars=got_chars,
        conservation_pass=ok,
        per_part={p: {"nodes": v[0], "chars": v[1]} for p, v in expected.items()},
        character_styles_defined=sum(
            1 for v in style_index.values() if v["type"] == "character"),
    )

    if not ok:
        raise ConservationError(report)

    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist(
        [], schema=pa.schema([(name, pa.null()) for name in RUN_SCHEMA_FIELDS]))
    return IngestResult(runs=table, conservation=report)


def write_runs_parquet(table: pa.Table, destination: Path) -> None:
    """Write an ingested run table to Parquet, zstd-compressed."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
