"""Dictionary-body segmentation: structural entry boundaries and IDs."""
from .dictionary import (
    SegmentationError,
    SegmentationReport,
    SegmentationResult,
    segment_dictionary,
    write_entries_parquet,
)

__all__ = [
    "SegmentationError",
    "SegmentationReport",
    "SegmentationResult",
    "segment_dictionary",
    "write_entries_parquet",
]
