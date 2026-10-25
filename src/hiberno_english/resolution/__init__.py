"""Deterministic post-generation resolution (rho): fills
`Region.resolved_name`/`.source_type` and `CrossReference.resolution`/
`.targets`/`.resolves` from source-side authority data, independent of
retrieval and model execution."""
from .deterministic import (
    DeterministicResolutionContext,
    DeterministicResolutionError,
    build_resolution_context,
    resolve_entry,
)

__all__ = [
    "DeterministicResolutionContext",
    "DeterministicResolutionError",
    "build_resolution_context",
    "resolve_entry",
]
