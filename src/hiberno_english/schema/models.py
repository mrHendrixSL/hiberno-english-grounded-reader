"""Pydantic schema for the structured lexical entry.

Every leaf value that could plausibly vary across independent readings is
wrapped in `Sourced[T]`, an envelope of {value, source_span, tier,
decided_by, authority_ref?, status}: not just fields checked against a
closed vocabulary, but ordinary content too, so that any point of
disagreement between readers can be traced back to where it came from and
how it was resolved.

`tier` and `status` are two different axes. `tier` records which settlement
rule produced a value (an authority span taken directly from the source, a
consensus span agreed across readers, or manual triage). `status` is the
terminal outcome for that value, which is not necessarily a success. Both
are `None` before any reading has been produced for a given field -- that
is the correct pre-extraction state, not a missing value.

`SourceSpan.locator` deliberately never encodes a page number: the
underlying source material has multiple incompatible paginations across its
different reproductions, so locations are cited structurally (headword,
entry id, or a document-relative locator) instead.
"""
from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")

TerminalStatus = Literal[
    "SETTLED", "DECLARED", "OBSERVED_EXTENSION", "SOURCE_ERROR",
    "AMBIGUOUS", "OUT_OF_SCHEMA", "DEFERRED",
]
SettlementTier = Literal["authority_span", "consensus_span", "triage"]


class SourceSpan(BaseModel):
    """Where a value comes from in the source text. Never a page number."""
    locator: str                          # e.g. "word/document.xml#p2557"
    para_idx: int | None = None
    xpath: str | None = None


class Sourced(BaseModel, Generic[T]):
    """Per-value provenance envelope: {value, source_span, tier,
    decided_by, authority_ref?, status}. All fields are carried on every
    wrapped leaf. `tier`, `decided_by`, and `status` are `None` until a
    reading has been produced for this value -- that is the correct
    pre-extraction state, not an omission.
    """
    value: T
    source_span: SourceSpan | None = None
    tier: SettlementTier | None = None
    decided_by: str | list[str] | None = None
    authority_ref: str | None = None      # e.g. "pos_labels:n." -- closed fields only
    status: TerminalStatus | None = None


class Pronunciation(BaseModel):
    ipa: Sourced[str]
    regional_qualifier: str | None = None
    sense_number: int | None = None


class Etymology(BaseModel):
    """`< Ir.` states a derivation; `cf. Ir` only draws a comparison. These
    are not the same claim at different confidence levels -- `cf.` is a
    distinct closed-vocabulary editorial abbreviation, not a language code,
    which is why `relation` is a separate field rather than folding the
    distinction into a second array."""
    # `None` when the source doesn't state a language marker for this
    # specific step -- e.g. one component of a chained/component etymology
    # printed with no code of its own (an eponym like "Robert Peel", an
    # all-caps root citation like "SHAG", an English-internal morpheme like
    # "mis-"). Source-unspecified, not "not applicable" -- confirmed with
    # the human reviewers against real cases (clitherer, gansey, peeler,
    # shagging, scheme, misfortunate, pannikin, pieceen, whelm, Irish
    # Sweepstakes) before loosening this field, not guessed.
    language_code: Sourced[str] | None = None   # closed vocabulary: language codes
    # `None` when the source prints only a bare marker (`< Ir.`) with no
    # etymon spelled out separately from the headword itself -- a real,
    # common source shape (headword IS the foreign word), not a missing
    # value to paper over. Readers vary in how they handle this case (RB
    # currently abstains from emitting the Etymology entry at all; LLM
    # readers are prompted to use `""`) -- this field accepts `None` so a
    # human adjudicator is never forced to invent a word that isn't there.
    etymon_form: str | None = None
    gloss: str | None = None
    # Which authority asserts this step, where the source attributes it:
    # some etymologies cite a specific reference work's claim about the
    # etymon rather than asserting it as the dictionary's own fact. Without
    # this field the etyma were kept but who claimed them was silently
    # dropped, flattening two competing published accounts into the
    # dictionary's own unattributed assertion.
    attributed_to: str | None = None
    chain_step: int = 1                   # 1 = immediate source; >1 = further back
    # `derivation` and `comparison` were the whole vocabulary, and the
    # source states further relations that were being lost or misfiled:
    #   diminutive_of  the etymon is itself a diminutive form of another
    #                  word, not a plain ancestor -- collapsing it into
    #                  `gloss` would assert the etymon MEANS "diminutive".
    #   variant_of     the etymon is recorded as a variant spelling/form of
    #                  another word, not a separate derivation.
    #   component      two or more elements COMBINE to form the headword,
    #                  not a chain of successive ancestors. Components
    #                  share a chain_step; they are not ordered.
    # `None` when the source states a language marker but no clear
    # derivation/comparison/etc. relation can be read from it yet --
    # confirmed against a real case (`clift`, "< E dial." alone) rather
    # than forcing a guessed relation onto an unclear step.
    relation: Literal["derivation", "comparison",
                      "diminutive_of", "variant_of", "component"] | None = None
    # A part-of-speech qualifier printed on the ETYMON itself, distinct
    # from the headword's own pos and from the gloss, which is a
    # translation.
    etymon_pos: str | None = None
    # Which numbered sense this step belongs to, for the entries where the
    # source gives each sense its own etymology. None means it scopes the
    # entry as a whole.
    sense_number: int | None = None


class Label(BaseModel):
    term: Sourced[str]                    # closed vocabulary: part-of-speech or register labels
    kind: Literal["pos", "register"]


class Region(BaseModel):
    """`(CODE, County)` -- the informant code is validated against a closed
    authority list; `resolved_name`/`source_type` are that authority's own
    descriptive fields, carried through rather than re-derived.

    `informant_code` is OPTIONAL because the source often prints a locality
    with no named informant at all. A county-only attribution is a real and
    common form of provenance in this dictionary, not a degenerate case of
    an informant citation; treating it as one previously forced the county
    itself to be duplicated into the informant-code field, asserting that a
    county is an informant.
    """
    informant_code: Sourced[str] | None = None   # closed vocabulary: registered informant/source codes
    county: str | None = None
    resolved_name: str | None = None
    source_type: Literal["person", "publication"] | None = None
    locus: str | None = None              # page/section within a cited glossary


class Attestation(BaseModel):
    """`citation_class` follows the source text's own printed typographic
    convention (double quotes = literary, single quotes = oral), not an
    inference from the nature of the cited work -- the two can and do
    diverge."""
    quote: str
    citation_class: Literal["literary", "oral"]
    source: str                           # informant_code | cited_work | reference_work
    locus: str | None = None              # locus WITHIN the cited work, not this document
    # What the source prints INSTEAD of an attribution -- a tag for what
    # kind of utterance this is (e.g. a proverb or a saying), or the
    # occasion on which it was said, rather than who said it. A genre tag
    # is not an attribution and never resolves to anyone.
    genre: str | None = None


class UsageNote(BaseModel):
    """A note the source prints that no other field holds.

    Reading a broad sample of these notes shows they are not one kind of
    thing: an origin claim with cited evidence, a named informant's remark,
    a fixed phrase with its own gloss, a translation of the adjacent
    citation, the occasion on which something was said, a bare
    bibliographic reference, and genuine free commentary.

    `kind` is deliberately optional and mostly unset. It is assigned only
    where the note's opening is unambiguous. Classifying the rest by
    surface pattern (a delimiter, a punctuation split) has repeatedly
    misfired on this kind of free text. An unset `kind` means "not yet
    determined", never "ordinary commentary".
    """
    text: Sourced[str]
    kind: Literal["origin_status", "attributed_comment", "phrase",
                  "gloss_of_citation", "occasion", "reference",
                  "variant_note"] | None = None
    attributed_to: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare(cls, v):
        """A bare string is an untyped note -- the common case for what a
        reader emits."""
        if isinstance(v, str):
            return {"text": {"value": v}}
        if isinstance(v, dict) and "value" in v and "text" not in v:
            return {"text": v}          # a bare Sourced[str]-shaped dict
        if isinstance(v, Sourced):
            return {"text": v}          # a Sourced instance constructed directly
        return v


class Gloss(BaseModel):
    """One gloss inside a sense, with what the source scopes to IT alone.

    The source routinely attributes part of a sense and not the rest -- a
    single sense can print several short glosses in sequence, each with its
    own informant or region, rather than one uniform attribution for the
    whole sense. Holding the whole run as one `definition` with the
    informants hoisted to sense level widens each attribution over material
    the source never assigned to it -- an over-assertion of provenance, and
    the harder kind to notice because nothing is missing and nothing is
    malformed.

    This is deliberately additive. `Sense.definition` keeps the printed run
    verbatim, because that is what conservation requires and because
    splitting it into separate senses would manufacture boundaries the
    source does not number. `glosses` records the division the source does
    make, and is empty wherever the sense is not subdivided.
    """
    text: Sourced[str]
    regions: list[Region] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)   # e.g. a register label on one gloss only


class Sense(BaseModel):
    """`number` is `None` for the large majority of entries, which print a
    single unnumbered sense -- forcing a fake `1` there would manufacture
    structure the source doesn't assert. Only a minority of entries print
    real 1./2./3. numbering.

    `definition_attribution` names the authority a definition is quoted
    from, where the source gives one. Without this field, the only carrier
    for that attribution was `Attestation.source`, so the definition had to
    be duplicated into a fabricated attestation to keep its provenance --
    which asserted that the named authority USED the word in a sentence,
    when they were in fact defining it. This field records who wrote the
    gloss, not that anyone was recorded saying the word.
    """
    number: int | None = None
    # `None` for a sense record that deliberately carries only a
    # quotation/attestation and no restated definition of its own (e.g. a
    # second illustrative quote kept as its own structural unit rather than
    # appended to the first sense's attestations) -- confirmed against real
    # cases (`bulling`, `equal`) with the human reviewers before loosening
    # this field, not guessed. Still required to be present (if filled) via
    # `Sourced`; `None` means "deliberately no definition text," not
    # "missing."
    definition: Sourced[str] | None = None
    definition_attribution: str | None = None
    glosses: list[Gloss] = Field(default_factory=list)
    labels: list[Label] = Field(default_factory=list)
    regions: list[Region] = Field(default_factory=list)
    attestations: list[Attestation] = Field(default_factory=list)


class CrossReference(BaseModel):
    """`type` distinguishes an entry that IS nothing but a redirect from an
    inline "See X" mentioned inside an entry that has its own content.
    `redirect` deliberately covers both typographic conventions the source
    uses for a pointer-only entry -- the distinction between them was a
    typographic accident, not a semantic one, so collapsing it here is
    intentional, not an oversight."""
    target: Sourced[str]
    type: Literal["redirect", "inline"]
    resolves: bool = False                # True once the target is confirmed to exist
    # `resolves` alone cannot distinguish the ways a pointer can succeed or
    # fail, and the difference matters for referential-integrity claims:
    #   unique         resolves to exactly one entry
    #   homonyms       the form is shared and the pointer does not say which
    #   subentry       the target is printed inside another entry, not as a
    #                  top-level lemma
    #   dangling       no referent anywhere in the corpus
    #   not_a_pointer  a bibliographic reference that the source's own
    #                  cross-reference typographic convention caused to be
    #                  typed as a cross-reference
    resolution: Literal["unique", "homonyms", "subentry",
                        "dangling", "not_a_pointer"] | None = None
    targets: list[str] = Field(default_factory=list)   # entry_ids the target hits


class Provenance(BaseModel):
    source_span: SourceSpan
    witness_id: str                       # digest identifying the exact source document revision
    decisions: list["Decision"] = Field(default_factory=list)


class Decision(BaseModel):
    """One append-only ledger entry recording a decision made about a
    specific field of this entry."""
    field_path: str
    status: TerminalStatus
    decided_by: str
    rationale: str


class VariantForm(BaseModel):
    """A form printed as an alternative to the headword, with what the
    source attaches to it specifically.

    A plain list of strings loses everything the source scopes to a
    PARTICULAR variant: which informant reported which spelling, a
    regional qualifier attached to just one variant, or a variant printed
    inside a numbered sense rather than at the head of the entry.

    The discriminator for whether an informant attaches to the variant or
    to the definition is positional and printed: an informant BEFORE the
    part-of-speech label belongs to the variant list, one after the gloss
    belongs to the gloss.
    """
    form: Sourced[str]
    pronunciations: list[Pronunciation] = Field(default_factory=list)
    qualifier: str | None = None          # a regional or dialectal qualifier on this variant
    regions: list[Region] = Field(default_factory=list)
    sense_number: int | None = None       # when printed inside a numbered sense

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_string(cls, v):
        """A bare string is a variant with nothing attached -- the common
        case for what a reader emits. Accepting it keeps plain-string data
        working while letting the richer shape carry a qualifier or an
        informant where the source gives one.
        """
        if isinstance(v, str):
            return {"form": {"value": v}}
        return v


class SubEntry(BaseModel):
    """A named item printed INSIDE another entry, with apparatus of its own.

    The source does this routinely, and it is not the same as a sense: a
    named form is introduced inside the body of a larger entry, with its
    own pronunciation, part of speech, gloss, informant, or etymology,
    rather than being a further sense of the same headword. With no node
    to hold them, such material was previously folded into the surrounding
    definition text or flattened into `usage_notes` as an untyped string.

    Deliberately NOT a recursive Entry: sub-entries in this source do not
    nest further, and modelling them as full entries would invite
    fabricating an entry_id and an entry_type the source never assigns.
    """
    form: Sourced[str]
    pronunciations: list[Pronunciation] = Field(default_factory=list)
    pos: list[Label] = Field(default_factory=list)
    senses: list[Sense] = Field(default_factory=list)
    etymology: list[Etymology] = Field(default_factory=list)
    usage_notes: list[UsageNote] = Field(default_factory=list)


class Entry(BaseModel):
    """`pos` and `Sense.labels` are both needed, because the source prints
    part of speech in two different scopes, and the difference is printed
    and unambiguous: sometimes a single part-of-speech marker applies once,
    before any numbered sense, scoping the whole headword; sometimes each
    numbered sense carries its own marker, because different senses really
    are different parts of speech.

    Carrying only sense-level labels forces an entry-level marker to be
    replicated onto every sense, which asserts a distribution the source
    does not print. Carrying only an entry-level `pos` loses the cases
    where a later sense genuinely is a different part of speech. `pos` is
    therefore an entry-level default; `Sense.labels` overrides it for the
    specific sense where the source prints its own marker.
    """
    entry_id: str
    entry_type: Literal["full", "cross_ref", "letter_header", "front", "back"]
    headword: Sourced[str]
    homonym_index: int | None = None
    pos: list[Label] = Field(default_factory=list)
    variant_forms: list[VariantForm] = Field(default_factory=list)
    pronunciations: list[Pronunciation] = Field(default_factory=list)
    etymology: list[Etymology] = Field(default_factory=list)
    senses: list[Sense] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)
    usage_notes: list[UsageNote] = Field(default_factory=list)
    sub_entries: list[SubEntry] = Field(default_factory=list)
    provenance: Provenance


Provenance.model_rebuild()
