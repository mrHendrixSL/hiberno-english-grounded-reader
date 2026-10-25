"""Structural precedent retrieval: Euclidean top-k lookup over a frozen
index of structural PCA coordinates."""
from .structural import (
    StructuralPrecedentIndex,
    StructuralRetrievalError,
    build_structural_precedent_index,
    load_frozen_structural_transform,
    retrieve_structural_precedents,
)

__all__ = [
    "StructuralPrecedentIndex",
    "StructuralRetrievalError",
    "build_structural_precedent_index",
    "retrieve_structural_precedents",
    "load_frozen_structural_transform",
]
