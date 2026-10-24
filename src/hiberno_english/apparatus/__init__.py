"""Dictionary apparatus / authority-table extraction."""
from .authority import (
    ApparatusError,
    ApparatusReport,
    ApparatusResult,
    extract_authorities,
    write_authority_tables,
)

__all__ = [
    "ApparatusError",
    "ApparatusReport",
    "ApparatusResult",
    "extract_authorities",
    "write_authority_tables",
]
