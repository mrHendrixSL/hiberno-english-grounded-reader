"""Field-group prompt construction: one shared template set used
identically for the bare and grounded reader conditions.

The ONLY permitted difference between a bare-condition prompt (no
precedents) and a grounded-condition prompt (precedents supplied) is
whether a precedent section is present -- never a different instruction, a
different response contract, or a different system prompt. This module
performs no retrieval, ranks nothing, and knows neither E390 membership
nor k: `build_field_group_requests` formats whatever precedent sequence
its caller supplies, in the order supplied, and makes no model call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from pydantic import BaseModel

from ...schema import Entry
from .contracts import FIELD_GROUPS, RESPONSE_MODELS


class ReaderPromptError(RuntimeError):
    """Raised when a field-group request cannot be built from the supplied
    source entry, authority context, or precedents."""


@dataclass(frozen=True)
class FieldGroupRequest:
    """One field-group's textual generation request. Carries no model
    identity and no decoding parameters -- those belong to the future
    frozen execution configuration, not to prompt construction."""

    group: str
    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]


SYSTEM_PROMPT = (
    "You extract structured data from ONE entry of a printed "
    "Hiberno-English dictionary. Read only what is printed. If the entry "
    "does not contain the thing being asked for, return an EMPTY list or "
    "null field rather than guessing or inventing content -- an empty "
    "answer is correct and expected for many entries, and for many of the "
    "fields asked about it is the common case. Never translate, "
    "normalise, or correct the source text; copy it exactly, including "
    "spelling and diacritics."
)


def _term_line(label: str, terms: object) -> str:
    if not terms:
        return ""
    return f"Valid {label}: " + ", ".join(str(t) for t in terms) + "\n\n"


def _instruction(group: str, source_entry: str,
                 authority_context: Mapping[str, object]) -> str:
    if group == "head_pron":
        return (
            f"Entry text:\n{source_entry}\n\n"
            + _term_line("part-of-speech labels", authority_context.get("pos_labels"))
            + "Extract the HEADWORD (without any homonym digit that "
            "follows it). 'ipa' values are printed BETWEEN SLASHES "
            "('/example/') -- report the transcription only, WITHOUT the "
            "two slash characters. 'pos' is the part-of-speech label(s) "
            "printed ONCE, before sense 1 (e.g. 'entryword /example/ n., "
            "adj. 1. First meaning ... 2. Second meaning ...' -> pos is "
            "[n., adj.]) -- use ONLY terms from the valid list, omit "
            "anything not exactly in it. If POS is instead printed "
            "separately after EACH numbered sense (e.g. 'entryword "
            "1. v., ... 2. n., ...'), leave 'pos' empty here -- that "
            "belongs to the senses group instead. 'pronunciations' is "
            "every /.../ block printed at the head, 'regional_qualifier' "
            "is a variety note printed with it if any (e.g. 'in a named "
            "regional variety'), 'sense_number' is null unless a "
            "specific pronunciation is explicitly tied to one numbered "
            "sense (rare). 'variant_forms' is ONLY a spelling explicitly "
            "introduced by the literal printed word 'also' (or, more "
            "rarely, 'or') immediately before it -- NOT a dialect "
            "spelling that merely appears inside a quoted example "
            "sentence elsewhere in the entry (a quoted spelling inside "
            "an attestation is evidence for a sense's attestation, never "
            "a variant_forms item). Two shapes of the SAME 'also' "
            "construction: before the part of speech (scoping the whole "
            "lemma, sense_number null: 'headword also variant adj.' -> "
            "form 'variant') OR inside one numbered sense's own text "
            "(scoping just that sense, sense_number set to it: sense 2 "
            "reads '2. Also \"variant-a\", \"variant-b\" (XX, "
            "Countyname), ...' -> two items, form 'variant-a' and form "
            "'variant-b', BOTH sense_number 2, WITH the region (XX, "
            "Countyname) printed alongside them). Report the form "
            "itself only -- WITHOUT the word 'also', without quotation "
            "marks, and without any other punctuation. 'qualifier' is a "
            "variety/register note printed with the variant (e.g. '(in "
            "a named regional variety)'). Do not confuse a variant "
            "SPELLING with a pronunciation -- never put an IPA "
            "transcription between slashes here; IPA belongs in "
            "'pronunciations' instead. Most entries have no "
            "variant_forms at all -- when in doubt, leave it empty "
            "rather than pull in a quoted spelling from elsewhere.")
    if group == "etymology":
        return (
            f"Entry text:\n{source_entry}\n\n"
            + _term_line("language codes", authority_context.get("language_codes"))
            + "Extract every ETYMOLOGY the entry states, one item per "
            "stated relation. Five relations, by their printed marker:\n"
            "  '< X'            -> relation \"derivation\"\n"
            "  'cf. X'          -> relation \"comparison\"\n"
            "  '(dimin. of X)'  -> relation \"diminutive_of\" -- X is "
            "often a Hiberno-English headword in capitals, not a "
            "foreign form\n"
            "  '(variant of X)' -> relation \"variant_of\"\n"
            "  elements joined by comma or '+' that COMBINE to form the "
            "word (e.g. 'part-one, meaning-one + part-two, meaning-two') "
            "-> relation \"component\", one item per element, all "
            "sharing the SAME chain_step\n"
            "'chain_step' starts at 1; a further-back link in a stated "
            "chain ('< X < Y') increments it; component siblings share "
            "one step. 'language_code' MUST be one of the valid codes "
            "above, exactly as printed, for derivation/comparison/"
            "component only (diminutive_of/variant_of etyma are often "
            "Hiberno-English, not foreign -- leave language_code as the "
            "closest marker printed, or the dictionary's own dialect "
            "marker if none is stated). 'etymon_form' is the "
            "word/phrase given, not a translation -- empty string if "
            "the marker has no form printed after it. 'gloss' is the "
            "English meaning given, if any. 'etymon_pos', "
            "'sense_number', 'attributed_to' are rarely printed -- "
            "leave null unless explicitly stated. If NONE of '<', "
            "'cf.', '(dimin. of', '(variant of', or a component join "
            "appear anywhere in the text, this entry has NO etymology "
            "-- return an empty list. Do not infer an etymology from "
            "spelling or context.")
    if group == "senses":
        return (
            f"Entry text:\n{source_entry}\n\n"
            + _term_line("part-of-speech labels", authority_context.get("pos_labels"))
            + _term_line("register labels", authority_context.get("register_labels"))
            + "Extract every SENSE (distinct meaning). Most entries have "
            "exactly one; only split on explicit numbering (\"1. ... "
            "2. ...\"), else 'number' is null. 'definition' is the full "
            "definition text verbatim. 'definition_attribution' is a "
            "source named directly with the definition itself (e.g. "
            "'(XYZ)' right after a quoted definition) -- distinct from "
            "an attestation, and rare; leave null unless clearly this "
            "shape. 'glosses' is ONLY for a definition that packs "
            "multiple attributed clauses together (e.g. 'meaning-a (XX, "
            "Countyname); meaning-b' -> two glosses, one scoped to "
            "XX/Countyname, one not) -- leave empty for an ordinary "
            "single-clause definition, which is the common case. "
            "'labels' are pos/register terms printed AFTER a specific "
            "numbered sense (not the entry-level pos already asked for "
            "in the head_pron group) -- ONLY from the two valid lists "
            "above, exactly as printed. 'regions' are informant "
            "citations '(CODE, County)' -- informant_code is null for a "
            "bare locality with no code (e.g. just '(Countyname)'), "
            "county is what's printed either way, 'locus' is a "
            "sub-location within the county if given (rare). "
            "'attestations' are quoted examples: 'quote' verbatim; "
            "citation_class \"literary\" for DOUBLE curly quotes, "
            "\"oral\" for SINGLE curly quotes -- the book's own "
            "convention, no other signal; 'source' is the author/work "
            "or informant code; 'locus' a literary page/line if given; "
            "'genre' is a parenthetical marker with NO named source "
            "(e.g. a proverb or saying marker) -- when genre is set, "
            "source is \"\".")
    if group == "cross_references":
        return (
            f"Entry text:\n{source_entry}\n\n"
            "Extract every CROSS-REFERENCE to another headword in this "
            "dictionary. STRICT RULE: exists ONLY where 'see'/'See' is "
            "printed immediately before the target. 'target' is the "
            "capitalised word(s) right after that 'see'/'See', exactly "
            "as printed. Do NOT extract from a language code, a "
            "region/informant code, a person's name or reference-work "
            "author, or any other capitalised word not immediately "
            "preceded by 'see'/'See' -- being capitalised alone is "
            "never enough. 'type' is \"redirect\" if the entry is "
            "nothing but a pointer, \"inline\" if it also defines its "
            "own headword. Most entries have none -- return an empty "
            "list.")
    if group == "usage_notes":
        return (
            f"Entry text:\n{source_entry}\n\n"
            "Extract any USAGE NOTE: a general remark about how/when/"
            "where the word is used that is NOT itself a definition, an "
            "etymology, or a quoted example. 'attributed_to' is a "
            "source named for the remark itself, if any. Most entries "
            "have none -- return an empty list. Do not invent a note "
            "from the definition text.")
    raise ReaderPromptError(f"unknown field group {group!r}")


def _region_dict(r) -> dict:
    return {
        "informant_code": r.informant_code.value if r.informant_code else None,
        "county": r.county,
        "locus": r.locus,
    }


def _label_dict(l) -> dict:
    return {"term": l.term.value, "kind": l.kind}


def _precedent_value(entry: Entry, group: str) -> object:
    """The precedent's OWN reading for `group` only -- never the whole
    Entry, and never a resolution field (a precedent must not show the
    model the answer to a question it is not being asked)."""
    if group == "head_pron":
        return {
            "headword": entry.headword.value,
            "pos": [_label_dict(l) for l in entry.pos],
            "pronunciations": [
                {"ipa": p.ipa.value, "regional_qualifier": p.regional_qualifier,
                 "sense_number": p.sense_number}
                for p in entry.pronunciations],
            "variant_forms": [
                {"form": vf.form.value, "qualifier": vf.qualifier,
                 "sense_number": vf.sense_number,
                 "regions": [_region_dict(r) for r in vf.regions],
                 "pronunciations": [
                     {"ipa": p.ipa.value, "regional_qualifier": p.regional_qualifier,
                      "sense_number": p.sense_number}
                     for p in vf.pronunciations]}
                for vf in entry.variant_forms],
        }
    if group == "etymology":
        return [
            {"language_code": t.language_code.value, "etymon_form": t.etymon_form,
             "gloss": t.gloss, "relation": t.relation, "chain_step": t.chain_step,
             "etymon_pos": t.etymon_pos, "sense_number": t.sense_number,
             "attributed_to": t.attributed_to}
            for t in entry.etymology]
    if group == "senses":
        return [
            {"number": s.number, "definition": s.definition.value,
             "definition_attribution": s.definition_attribution,
             "glosses": [{"text": g.text.value,
                          "regions": [_region_dict(r) for r in g.regions],
                          "labels": [_label_dict(l) for l in g.labels]}
                         for g in s.glosses],
             "labels": [_label_dict(l) for l in s.labels],
             "regions": [_region_dict(r) for r in s.regions],
             "attestations": [
                 {"quote": a.quote, "citation_class": a.citation_class,
                  "source": a.source, "locus": a.locus, "genre": a.genre}
                 for a in s.attestations]}
            for s in entry.senses]
    if group == "cross_references":
        return [{"target": x.target.value, "type": x.type} for x in entry.cross_references]
    if group == "usage_notes":
        return [{"text": n.text.value, "attributed_to": n.attributed_to}
                for n in entry.usage_notes]
    raise ReaderPromptError(f"unknown field group {group!r}")


def _precedent_section(group: str, precedents: Sequence[Entry]) -> str:
    if not precedents:
        return ""
    blocks = [json.dumps(_precedent_value(p, group), ensure_ascii=False)
              for p in precedents]
    return ("Worked examples from similar, already-checked dictionary "
            "entries (for calibration only -- this entry's own text is "
            "what you must read):\n" + "\n".join(blocks) + "\n\n")


def build_field_group_requests(
    *,
    source_entry: str,
    authority_context: Mapping[str, object],
    precedents: Sequence[Entry] = (),
) -> tuple[FieldGroupRequest, ...]:
    """Build exactly five FieldGroupRequests, one per FIELD_GROUPS entry,
    in that order. Performs NO model call.

    `source_entry` is used verbatim -- never normalised, rewritten, or
    used to perform retrieval. `authority_context` supplies whatever
    closed-vocabulary term lists a group's instructions reference (keys
    "pos_labels", "register_labels", "language_codes"); a missing key
    simply omits that constraint line rather than failing.

    `precedents` is used exactly as supplied, in the order supplied --
    this function does not retrieve, rank, sort, or inspect precedents,
    and knows neither E390 membership nor k. Empty (the bare-reader
    condition) omits the precedent section for every group; a non-empty
    sequence (the grounded-reader condition) is the only permitted
    difference from the bare condition's prompts.
    """
    return tuple(
        FieldGroupRequest(
            group=group,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=(_precedent_section(group, precedents)
                        + _instruction(group, source_entry, authority_context)),
            response_model=RESPONSE_MODELS[group],
        )
        for group in FIELD_GROUPS
    )
