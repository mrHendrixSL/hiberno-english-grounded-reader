"""Response contracts for the five field-group generation calls, and the
typed container that requires exactly one validated response per group.

These five response models are the RAW extraction contract: what the
language model is asked to transcribe from the printed entry, never a
value a later deterministic pass computes. In particular, none of them
carries a resolution field -- the public schema's
`CrossReference.resolution`/`.targets`/`.resolves` or
`Region.resolved_name`/`.source_type` -- so a generated response cannot
even represent a resolved identity, let alone fabricate one; deterministic
resolution fills those in afterward, outside this module.

Every model below that may be populated from raw model output forbids
unexpected JSON fields (`_StrictResponseModel`, `extra="forbid"`): this
is the boundary between untrusted generation output and lexical assembly,
so a generated `resolution`/`targets`/`resolves`/`resolved_name`/
`source_type` -- or any other out-of-contract key -- is rejected outright
rather than silently discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FIELD_GROUPS: tuple[str, ...] = (
    "head_pron", "etymology", "senses", "cross_references", "usage_notes")


class ReaderContractError(RuntimeError):
    """Raised when a field-group response, or the assembled set of five
    responses, does not satisfy the generation contract."""


class _StrictResponseModel(BaseModel):
    """Internal base for every model-facing response shape: forbids any
    JSON field not explicitly declared. Not part of the public API."""
    model_config = ConfigDict(extra="forbid")


# ── shared item shapes ───────────────────────────────────────────────────────
class LabelItem(_StrictResponseModel):
    term: str
    kind: Literal["pos", "register"]


class RegionItem(_StrictResponseModel):
    informant_code: str | None = None
    county: str | None = None
    locus: str | None = None


class PronunciationItem(_StrictResponseModel):
    ipa: str
    regional_qualifier: str | None = None
    sense_number: int | None = None


class VariantFormItem(_StrictResponseModel):
    form: str
    qualifier: str | None = None
    sense_number: int | None = None
    regions: list[RegionItem] = Field(default_factory=list)
    pronunciations: list[PronunciationItem] = Field(default_factory=list)


# ── head_pron: owns Entry.headword/pos/pronunciations/variant_forms ────────
class HeadPronResponse(_StrictResponseModel):
    """Does NOT own Entry.homonym_index: the source prints the homonym
    digit as document/segmentation structure, not as something a reader
    transcribes independently, so it is supplied at assembly as
    workflow/source metadata, never generated here."""
    headword: str
    pos: list[LabelItem] = Field(default_factory=list)
    pronunciations: list[PronunciationItem] = Field(default_factory=list)
    variant_forms: list[VariantFormItem] = Field(default_factory=list)


# ── etymology: owns Entry.etymology ─────────────────────────────────────────
class EtymologyItem(_StrictResponseModel):
    language_code: str
    etymon_form: str
    gloss: str | None = None
    relation: Literal["derivation", "comparison", "diminutive_of",
                      "variant_of", "component"]
    chain_step: int = 1
    etymon_pos: str | None = None
    sense_number: int | None = None
    attributed_to: str | None = None


class EtymologyResponse(_StrictResponseModel):
    etymology: list[EtymologyItem] = Field(default_factory=list)


# ── senses: owns Entry.senses ────────────────────────────────────────────────
class AttestationItem(_StrictResponseModel):
    quote: str
    citation_class: Literal["literary", "oral"]
    source: str
    locus: str | None = None
    genre: str | None = None


class GlossItem(_StrictResponseModel):
    text: str
    regions: list[RegionItem] = Field(default_factory=list)
    labels: list[LabelItem] = Field(default_factory=list)


class SenseItem(_StrictResponseModel):
    number: int | None = None
    definition: str
    definition_attribution: str | None = None
    glosses: list[GlossItem] = Field(default_factory=list)
    labels: list[LabelItem] = Field(default_factory=list)
    regions: list[RegionItem] = Field(default_factory=list)
    attestations: list[AttestationItem] = Field(default_factory=list)


class SensesResponse(_StrictResponseModel):
    senses: list[SenseItem] = Field(default_factory=list)


# ── cross_references: owns Entry.cross_references ───────────────────────────
class CrossRefItem(_StrictResponseModel):
    """No `resolution`/`targets`/`resolves`: whether a target actually
    resolves to another entry is computed deterministically, never
    generated. `extra="forbid"` (via `_StrictResponseModel`) is what
    actually enforces this against raw model output -- these three keys
    being merely absent from the field list would otherwise be silently
    ignored, not rejected."""
    target: str
    type: Literal["redirect", "inline"]


class CrossReferencesResponse(_StrictResponseModel):
    cross_references: list[CrossRefItem] = Field(default_factory=list)


# ── usage_notes: owns Entry.usage_notes ──────────────────────────────────────
class UsageNoteItem(_StrictResponseModel):
    """No `kind`: the public schema's UsageNote.kind is deliberately left
    unassigned by generation -- see UsageNote's own docstring."""
    text: str
    attributed_to: str | None = None


class UsageNotesResponse(_StrictResponseModel):
    usage_notes: list[UsageNoteItem] = Field(default_factory=list)


RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "head_pron": HeadPronResponse,
    "etymology": EtymologyResponse,
    "senses": SensesResponse,
    "cross_references": CrossReferencesResponse,
    "usage_notes": UsageNotesResponse,
}


@dataclass(frozen=True)
class FieldGroupOutputs:
    """Exactly one validated response per field group -- never an
    unvalidated arbitrary dict. Field ownership across the five groups is
    disjoint (see readers.llm.assembly.FIELD_OWNERSHIP), so assembly never
    has to choose between competing values for the same field."""

    head_pron: HeadPronResponse
    etymology: EtymologyResponse
    senses: SensesResponse
    cross_references: CrossReferencesResponse
    usage_notes: UsageNotesResponse

    def __post_init__(self) -> None:
        for name, cls in RESPONSE_MODELS.items():
            value = getattr(self, name)
            if not isinstance(value, cls):
                raise ReaderContractError(
                    f"FieldGroupOutputs.{name} must be a {cls.__name__}, "
                    f"got {type(value).__name__}")
