"""The fixed clustering/retrieval projection phi_C and the frozen structural
transform T.

phi (`structural.FEATURE_COLUMNS`) is the complete source-side structural
feature vector. phi_C -- `CLUSTER_FEATURES` below -- is the fixed subset of
phi's count-valued features that E390's bulk clustering and the
grounded-reader's structural precedent retrieval both project through the
SAME frozen transform T. There is exactly one public implementation of T;
neither consumer refits it or defines its own variant.

T is: log1p on each retained count, then a StandardScaler (sklearn
defaults), then PCA retaining enough components to explain a chosen
fraction of variance. T is fit once, on a caller-supplied fitting
population, and frozen; applying it afterwards to any other table --
E390 precedents, a future query entry, H, or the full corpus D -- never
refits it.

This module owns feature transformation ONLY. It has no opinion about which
rows belong in a fitting population: no hard-stratum predicate
(`n_sense_num`, `has_homonym`, `is_pointer`, `is_cross_ref`) is read or
evaluated here. Deciding that a table is "the bulk population" is the
sample-design module's job; this module only ever transforms whatever
table it is handed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .structural import BOOL_FEATURES, COUNT_FEATURES

# ── phi_C: the fixed clustering/retrieval projection of phi ────────────────
# Excluded: n_sense_num, because it instead defines E390's take-all
# multi-sense stratum rather than being a clustering input; n_smallcaps_run
# and n_allcaps_run, because the source's own typography never populates
# them (see structural.py). Never manually re-listed: phi_C is derived from
# COUNT_FEATURES plus this explicit exclusion set, so it moves automatically
# if COUNT_FEATURES' declared order ever changes -- but the gates below stop
# an accidental upstream change from silently altering phi_C's membership or
# dimensionality instead of being caught.
_EXCLUDED_FROM_PHI_C: tuple[str, ...] = (
    "n_sense_num", "n_smallcaps_run", "n_allcaps_run")

if not set(_EXCLUDED_FROM_PHI_C) <= set(COUNT_FEATURES):
    raise RuntimeError(
        "structural feature contract changed: the phi_C exclusion set "
        f"{_EXCLUDED_FROM_PHI_C} is no longer a subset of COUNT_FEATURES "
        f"({COUNT_FEATURES!r}) -- phi_C cannot be safely derived from a "
        "changed upstream contract.")

CLUSTER_FEATURES: tuple[str, ...] = tuple(
    c for c in COUNT_FEATURES if c not in _EXCLUDED_FROM_PHI_C)

if len(CLUSTER_FEATURES) != len(COUNT_FEATURES) - len(_EXCLUDED_FROM_PHI_C):
    raise RuntimeError(
        "internal error deriving CLUSTER_FEATURES from COUNT_FEATURES")

if len(CLUSTER_FEATURES) != 25:
    raise RuntimeError(
        "phi_C contract violated: expected exactly 25 clustering/retrieval "
        f"features, derived {len(CLUSTER_FEATURES)} from the current "
        "COUNT_FEATURES/exclusion set -- the upstream structural feature "
        "declaration has changed; phi_C's fixed dimensionality can no "
        "longer be silently assumed.")

if set(CLUSTER_FEATURES) & set(BOOL_FEATURES):
    raise RuntimeError(
        "phi_C contract violated: overlaps BOOL_FEATURES -- no boolean "
        "feature of phi may be a clustering/retrieval input.")


class StructuralTransformError(RuntimeError):
    """Raised when fitting or applying the structural transform T cannot
    proceed on trustworthy input, or when a StructuralTransform's own
    frozen state is internally inconsistent. No partial or
    silently-incomplete transform or projection is ever returned alongside
    this exception.
    """


@dataclass(frozen=True)
class StructuralTransform:
    """The frozen, fitted state of T: log1p -> StandardScaler -> PCA.

    Every field is a plain, JSON-serialisable value (strings, ints, floats,
    and tuples of these) -- never a live sklearn estimator -- so this state
    can be inspected, compared, and later persisted without needing
    sklearn to read it back. `apply_structural_transform` reads only these
    fields; it never reconstructs or refits a scaler or a PCA object.
    """

    feature_names: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    n_components: int
    seed: int
    variance_threshold: float

    def __post_init__(self) -> None:
        p = len(self.feature_names)
        if len(self.scaler_mean) != p or len(self.scaler_scale) != p:
            raise StructuralTransformError(
                "malformed StructuralTransform: scaler_mean/scaler_scale "
                f"must each have length {p} (one per feature_name), got "
                f"{len(self.scaler_mean)}/{len(self.scaler_scale)}")
        if len(self.pca_mean) != p:
            raise StructuralTransformError(
                "malformed StructuralTransform: pca_mean must have length "
                f"{p} (one per feature_name), got {len(self.pca_mean)}")
        if self.n_components < 1:
            raise StructuralTransformError(
                "malformed StructuralTransform: n_components must be >= 1, "
                f"got {self.n_components}")
        if len(self.pca_components) != self.n_components:
            raise StructuralTransformError(
                "malformed StructuralTransform: pca_components must have "
                f"{self.n_components} rows (one per retained component), "
                f"got {len(self.pca_components)}")
        if any(len(row) != p for row in self.pca_components):
            raise StructuralTransformError(
                "malformed StructuralTransform: every pca_components row "
                f"must have length {p} (one weight per feature)")
        if len(self.explained_variance_ratio) != self.n_components:
            raise StructuralTransformError(
                "malformed StructuralTransform: explained_variance_ratio "
                f"must have length {self.n_components}, got "
                f"{len(self.explained_variance_ratio)}")

    def to_dict(self) -> dict:
        """A plain JSON-compatible representation of this frozen state --
        lists, ints, floats, strings only. Never a pickle."""
        return {
            "feature_names": list(self.feature_names),
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "pca_mean": list(self.pca_mean),
            "pca_components": [list(row) for row in self.pca_components],
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "n_components": self.n_components,
            "seed": self.seed,
            "variance_threshold": self.variance_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuralTransform":
        """Reconstruct a StructuralTransform from `to_dict`'s output.
        Raises StructuralTransformError on a missing key or a shape that
        fails the same internal-consistency checks the constructor
        enforces on every StructuralTransform."""
        try:
            return cls(
                feature_names=tuple(data["feature_names"]),
                scaler_mean=tuple(float(v) for v in data["scaler_mean"]),
                scaler_scale=tuple(float(v) for v in data["scaler_scale"]),
                pca_mean=tuple(float(v) for v in data["pca_mean"]),
                pca_components=tuple(
                    tuple(float(v) for v in row)
                    for row in data["pca_components"]),
                explained_variance_ratio=tuple(
                    float(v) for v in data["explained_variance_ratio"]),
                n_components=int(data["n_components"]),
                seed=int(data["seed"]),
                variance_threshold=float(data["variance_threshold"]),
            )
        except KeyError as exc:
            raise StructuralTransformError(
                f"malformed serialised StructuralTransform: missing key {exc}"
            ) from exc


def _extract_matrix(features: pa.Table, *, context: str) -> tuple[list, np.ndarray]:
    """Validate `features` against the phi_C contract and return
    (entry_id values in input row order, a float64 matrix of
    CLUSTER_FEATURES columns in canonical order, same row order).

    Never reads any column outside entry_id + CLUSTER_FEATURES: no
    BOOL_FEATURES column, and no hard-stratum predicate, has any way to
    reach or influence T through this function.
    """
    schema_names = set(features.schema.names)
    if "entry_id" not in schema_names:
        raise StructuralTransformError(
            f"cannot {context} a structural transform: input table has no "
            "'entry_id' column")

    missing = [c for c in CLUSTER_FEATURES if c not in schema_names]
    if missing:
        raise StructuralTransformError(
            f"cannot {context} a structural transform: input table is "
            f"missing phi_C column(s) {missing}")

    rows = features.to_pylist()
    ids = [r["entry_id"] for r in rows]

    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    if dup_ids:
        raise StructuralTransformError(
            f"cannot {context} a structural transform: duplicate "
            f"entry_id(s) {dup_ids}")

    matrix: list[list[float]] = []
    for r in rows:
        row_values: list[float] = []
        for c in CLUSTER_FEATURES:
            v = r[c]
            if v is None:
                raise StructuralTransformError(
                    f"cannot {context} a structural transform: entry "
                    f"{r['entry_id']!r} has a null value for feature {c!r}")
            fv = float(v)
            if not math.isfinite(fv):
                raise StructuralTransformError(
                    f"cannot {context} a structural transform: entry "
                    f"{r['entry_id']!r} has a non-finite value for "
                    f"feature {c!r}: {fv!r}")
            if fv < 0:
                raise StructuralTransformError(
                    f"cannot {context} a structural transform: entry "
                    f"{r['entry_id']!r} has a negative value for count "
                    f"feature {c!r}: {fv!r} -- log1p is not a valid "
                    "transform for a negative count")
            row_values.append(fv)
        matrix.append(row_values)

    return ids, np.array(matrix, dtype=np.float64)


def fit_structural_transform(
    features: pa.Table,
    *,
    seed: int = 20260821,
    variance_threshold: float = 0.90,
) -> StructuralTransform:
    """Fit T = log1p -> StandardScaler -> PCA on `features`, the fitting
    population, and return it frozen.

    `features` is used exactly as supplied: this function makes no
    judgement about which rows belong in it. Constructing the E390-design
    bulk population (excluding the hard strata) is the sample-design
    module's responsibility, not this one's -- keeping feature
    transformation strictly separate from sample-stratum assignment.

    Raises StructuralTransformError if `features` is empty or fails the
    phi_C input contract (see `_extract_matrix`).
    """
    # Imported here, not at module level: fitting is the ONLY operation
    # in this module that needs sklearn at all -- `StructuralTransform`
    # and `apply_structural_transform` are pure NumPy over already-fitted
    # plain-float state (see their own docstrings), so a held-out/
    # production consumer that only ever APPLIES an already-frozen T
    # never needs sklearn importable at all. Pure import-timing change;
    # fits byte-for-byte the same PCA/StandardScaler as before.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if features.num_rows == 0:
        raise StructuralTransformError(
            "cannot fit a structural transform on an empty table")

    _, X_raw = _extract_matrix(features, context="fit")

    X = np.log1p(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold, random_state=seed)
    pca.fit(X_scaled)

    return StructuralTransform(
        feature_names=CLUSTER_FEATURES,
        scaler_mean=tuple(float(v) for v in scaler.mean_),
        scaler_scale=tuple(float(v) for v in scaler.scale_),
        pca_mean=tuple(float(v) for v in pca.mean_),
        pca_components=tuple(
            tuple(float(v) for v in row) for row in pca.components_),
        explained_variance_ratio=tuple(
            float(v) for v in pca.explained_variance_ratio_),
        n_components=int(pca.components_.shape[0]),
        seed=seed,
        variance_threshold=variance_threshold,
    )


def apply_structural_transform(
    transform: StructuralTransform,
    features: pa.Table,
) -> pa.Table:
    """Project `features` through the already-fitted `transform`.

    Reads only `transform`'s frozen arrays -- never calls
    StandardScaler.fit / PCA.fit / fit_transform, and never reconstructs a
    live sklearn estimator. Applying T here can never alter it, so the
    same StructuralTransform may be applied repeatedly to E390 precedents,
    new query entries, H, or D without ever refitting.

    Returns a table of `entry_id`, `z_0`, ..., `z_{n_components-1}`, in
    that column order, with rows in `features`' input order exactly.

    Raises StructuralTransformError if `transform.feature_names` does not
    match the current phi_C contract, or if `features` fails the phi_C
    input contract (see `_extract_matrix`).
    """
    if transform.feature_names != CLUSTER_FEATURES:
        raise StructuralTransformError(
            "cannot apply this StructuralTransform: its feature_names do "
            "not match the current phi_C contract (CLUSTER_FEATURES) -- "
            "refusing to apply a transform fit under a different "
            f"structural feature declaration. transform.feature_names="
            f"{transform.feature_names!r}, current CLUSTER_FEATURES="
            f"{CLUSTER_FEATURES!r}")

    ids, X_raw = _extract_matrix(features, context="apply")

    X = np.log1p(X_raw)
    mean = np.array(transform.scaler_mean, dtype=np.float64)
    scale = np.array(transform.scaler_scale, dtype=np.float64)
    X_scaled = (X - mean) / scale

    pca_mean = np.array(transform.pca_mean, dtype=np.float64)
    components = np.array(transform.pca_components, dtype=np.float64)
    Z = (X_scaled - pca_mean) @ components.T

    columns: dict[str, list] = {"entry_id": ids}
    for j in range(Z.shape[1]):
        columns[f"z_{j}"] = Z[:, j].tolist()
    return pa.Table.from_pydict(columns)
