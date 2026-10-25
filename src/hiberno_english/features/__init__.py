"""Canonical source-side structural feature map (phi) and the fixed
clustering/retrieval projection phi_C with its frozen transform T."""
from .structural import (
    BOOL_FEATURES,
    COUNT_FEATURES,
    FEATURE_COLUMNS,
    StructuralFeatureError,
    build_structural_features,
)
from .transform import (
    CLUSTER_FEATURES,
    StructuralTransform,
    StructuralTransformError,
    apply_structural_transform,
    fit_structural_transform,
)

__all__ = [
    "BOOL_FEATURES",
    "COUNT_FEATURES",
    "FEATURE_COLUMNS",
    "StructuralFeatureError",
    "build_structural_features",
    "CLUSTER_FEATURES",
    "StructuralTransform",
    "StructuralTransformError",
    "apply_structural_transform",
    "fit_structural_transform",
]
