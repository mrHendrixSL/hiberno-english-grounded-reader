"""Source ingestion: lossless OOXML -> typed run table."""
from .ooxml import (
    ConservationError,
    ConservationReport,
    IngestResult,
    ingest_docx,
    write_runs_parquet,
)

__all__ = [
    "ConservationError",
    "ConservationReport",
    "IngestResult",
    "ingest_docx",
    "write_runs_parquet",
]
