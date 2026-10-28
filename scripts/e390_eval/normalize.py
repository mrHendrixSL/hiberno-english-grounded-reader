"""Candidate string-normalization profiles for the E390 scoring
sensitivity audit (Phase 2). Every profile is a pure, deterministic
`str -> str` function; none of them do fuzzy matching, edit distance,
embeddings, stemming, lemmatization, token overlap, or any form of
semantic similarity.

Only `N0`/`N1`/`N2` are candidates for adoption as an actual scoring
mode. `N3`/`N4` exist to be AUDITED (quantify their effect) -- see
`audit_normalization.py` -- and must never be adopted as the default or
silently substituted for the exact metric; whether they are exposed at
all as an explicit, clearly-labelled additional profile is a decision
made in the audit report, not here.

Applied only to string leaf values -- a non-string leaf (bool, int, an
enum literal) is never touched, by construction of `apply_profile`.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Mapping

# ── N2: conservative whitespace canonicalisation ────────────────────────────
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _canonicalize_whitespace(s: str) -> str:
    """Every character `str.isspace()` agrees is whitespace (this
    includes ordinary space/tab/CR/LF and NBSP U+00A0, which Python's
    Unicode whitespace category already covers) becomes a plain ASCII
    space; runs of whitespace collapse to one; leading/trailing
    whitespace is stripped. Never touches a non-whitespace character."""
    folded = "".join(" " if ch.isspace() else ch for ch in s)
    return _WHITESPACE_RUN_RE.sub(" ", folded).strip()


# ── N4: typographic-punctuation equivalence (audited, not adopted) ─────────
# Small, explicit, documented substitution tables -- never a general
# Unicode confusable-folding pass. Each entry maps a variant codepoint to
# one canonical representative.
QUOTE_EQUIVALENCE_MAP: Mapping[str, str] = {
    "‘": "'", "’": "'", "′": "'",   # ' ' ′ -> '
    "“": '"', "”": '"', "″": '"',   # " " ″ -> "
}
DASH_EQUIVALENCE_MAP: Mapping[str, str] = {
    "–": "-", "—": "-", "−": "-",   # – — − -> -
}
_TYPOGRAPHIC_MAP: dict[str, str] = {**QUOTE_EQUIVALENCE_MAP, **DASH_EQUIVALENCE_MAP}


def _translate(s: str, table: Mapping[str, str]) -> str:
    for src, dst in table.items():
        s = s.replace(src, dst)
    return s


# ── the five profiles ────────────────────────────────────────────────────────
def n0_exact(s: str) -> str:
    """Identity -- the current, only-ever-shipped behaviour."""
    return s


def n1_unicode_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def n2_conservative_text(s: str) -> str:
    return _canonicalize_whitespace(n1_unicode_nfc(s))


def n3_casefold(s: str) -> str:
    """AUDIT ONLY. Not adopted as a scoring profile without explicit,
    separate approval -- see contract.py/the audit report for why."""
    return n2_conservative_text(s).casefold()


def n4_typographic_punctuation(s: str) -> str:
    """AUDIT ONLY. Not adopted as a scoring profile without explicit,
    separate approval."""
    return _translate(n2_conservative_text(s), _TYPOGRAPHIC_MAP)


# Diagnostic-only partial transforms, used solely to attribute an N4
# resolution to "quote-style" vs "dash-style" specifically (see
# audit_normalization.py's categorisation) -- never used for scoring.
def _n4_quote_only(s: str) -> str:
    return _translate(n2_conservative_text(s), QUOTE_EQUIVALENCE_MAP)


def _n4_dash_only(s: str) -> str:
    return _translate(n2_conservative_text(s), DASH_EQUIVALENCE_MAP)


NORMALIZATION_PROFILES: Mapping[str, Callable[[str], str]] = {
    "N0_exact": n0_exact,
    "N1_unicode_nfc": n1_unicode_nfc,
    "N2_conservative_text": n2_conservative_text,
    "N3_casefold": n3_casefold,
    "N4_typographic_punctuation": n4_typographic_punctuation,
}

# Profiles this build's audit and design discussion concluded are safe to
# expose as an actual side-by-side scoring mode -- see the audit report.
# N3/N4 are deliberately absent: audited only, never adopted here.
ADOPTABLE_SCORING_PROFILES: Mapping[str, Callable[[str], str] | None] = {
    "exact": None,                          # None means "no normalization" -- the original, unchanged path
    "conservative_normalized": n2_conservative_text,
}


def apply_profile(values, profile: Callable[[str], str] | None):
    """Apply `profile` to every STRING value in `values`; every other
    value (bool, int, an enum literal, None already filtered upstream)
    passes through unchanged. `profile=None` is the identity -- the exact
    metric's own code path, never touched by this function at all when
    the caller omits it."""
    if profile is None:
        return list(values)
    return [profile(v) if isinstance(v, str) else v for v in values]
