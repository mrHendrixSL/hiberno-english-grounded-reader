"""E390 development evaluation: field-level precision/recall/F1 scoring
against the adjudicated E390 reference.

No scientific logic lives here: the adjudicated reference and a
candidate's own output are loaded and validated, never recomputed,
redrawn, or regenerated. E390 is development data, never a held-out
evaluation.

This is a partial port. The full original package also discovers and
validates LLM run directories on disk (`sources.py`, `common_support.py`)
and selects a routing winner among LLM candidates (`routing.py`) --
none of that is included here, since the real per-candidate LLM run
data those modules need no longer exists (see the pipeline notebook's
own note on this). Only the four self-contained computation modules
are ported: `contract` (the field-path classification every metric is
built from), `normalize` (string-normalization profiles), `scoring`
(precision/recall/F1), and `rollups` (aggregating per-path metrics into
per-field-group and targeted/full-schema views).
"""
