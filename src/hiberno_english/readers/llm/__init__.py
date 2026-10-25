"""Shared five-field-group LLM generation core, used identically by the
bare and grounded reader conditions, plus the shared Ollama execution layer
that runs an already-built request battery, plus the adjudicated-precedent
binding layer that turns retrieved precedent IDs into ordered Entry
sequences, plus the B/R/D/G factorial composition layer that is the first
layer in this package allowed to know that condition design. Deterministic
resolution mathematics still lives only in `hiberno_english.resolution`."""
from .assembly import FIELD_OWNERSHIP, assemble_entry
from .authority_context import LLMHasAuthorityContextError, build_llm_authority_context
from .contracts import (
    FIELD_GROUPS,
    CrossReferencesResponse,
    EtymologyResponse,
    FieldGroupOutputs,
    HeadPronResponse,
    ReaderContractError,
    SensesResponse,
    UsageNotesResponse,
)
from .execution import (
    CallTelemetry,
    ModelArtifactIdentity,
    ModelExecutionConfig,
    ModelExecutionError,
    OllamaRuntimeConfig,
    OllamaTransportProvider,
    TransportProvider,
    execute_field_group_request,
    execute_field_group_requests,
    model_execution_sha256,
    response_contract_sha256,
)
from .factorial import (
    FactorialReaderError,
    FactorialReaderOutputs,
    LLMEntryInput,
    compose_factorial_outputs,
    generate_llm_entry,
    run_factorial_entry,
)
from .precedents import (
    AdjudicatedPrecedentBank,
    PrecedentBindingError,
    QueryPrecedents,
    bind_retrieved_precedents,
    build_adjudicated_precedent_bank,
)
from .prompts import FieldGroupRequest, ReaderPromptError, build_field_group_requests

__all__ = [
    "FIELD_GROUPS",
    "FIELD_OWNERSHIP",
    "FieldGroupRequest",
    "FieldGroupOutputs",
    "HeadPronResponse",
    "EtymologyResponse",
    "SensesResponse",
    "CrossReferencesResponse",
    "UsageNotesResponse",
    "ReaderContractError",
    "ReaderPromptError",
    "build_field_group_requests",
    "assemble_entry",
    "ModelExecutionConfig",
    "OllamaRuntimeConfig",
    "ModelExecutionError",
    "model_execution_sha256",
    "execute_field_group_request",
    "execute_field_group_requests",
    "TransportProvider",
    "OllamaTransportProvider",
    "CallTelemetry",
    "ModelArtifactIdentity",
    "response_contract_sha256",
    "AdjudicatedPrecedentBank",
    "QueryPrecedents",
    "PrecedentBindingError",
    "build_adjudicated_precedent_bank",
    "bind_retrieved_precedents",
    "LLMEntryInput",
    "FactorialReaderOutputs",
    "FactorialReaderError",
    "generate_llm_entry",
    "compose_factorial_outputs",
    "run_factorial_entry",
    "LLMHasAuthorityContextError",
    "build_llm_authority_context",
]
