"""
Symbol index package for different storage backends.

This package provides abstract interfaces and concrete implementations
for symbol caching and querying.
"""

import logging
from pathlib import Path
from typing import Optional

from .symbol_index import SymbolIndex
from .kuzu_index import KuzuIndex
from .duckdb_index import DuckdbIndex,SymbolAppender, SingleAppender

log = logging.getLogger(__name__)

__all__ = [
    'SymbolIndex', 
    'KuzuIndex', 
    'DuckdbIndex', 
    'create_symbol_index',
    'SymbolAppender',
    'SingleAppender'
]


def create_symbol_index(
    base_dir: str,
    language_id: str,
    backend: str = "kuzu",
    db_name: str = "symbol_db"
) -> SymbolIndex:
    """
    Factory method to create a SymbolIndex instance.
    
    Storage organization:
    - Kuzu: Each language_id gets its own directory: {base_dir}/{language_id}/
    - DuckDB: Shared database file with schema-based isolation: {base_dir}/{db_name}.duckdb
    
    :param base_dir: Base directory for symbol caches
    :param language_id: Language identifier (e.g., "python", "javascript")
    :param backend: Backend type ("kuzu" or "duckdb"), defaults to "kuzu"
    :param db_name: Database name, defaults to "symbol_db"
    :return: SymbolIndex instance
    :raises ImportError: If the requested backend is not available
    :raises ValueError: If the backend type is not supported
    """
    backend = backend.lower()
    
    if backend == "kuzu":
        # Kuzu: Each language gets its own directory
        kuzu_dir = str(Path(base_dir) / language_id)
        try:
            index = KuzuIndex(kuzu_dir, db_name)
            log.info(f"Created Kùzu-based symbol index for {language_id} at {kuzu_dir}")
            return index
        except ImportError as e:
            raise ImportError(
                "Kùzu is required for 'kuzu' backend. Install it with: pip install kuzu"
            ) from e
    elif backend == "duckdb":
        # DuckDB: Shared database file, use language_id as schema name
        db_path = str(Path(base_dir) / f"{db_name}.duckdb")
        schema_name = language_id  # Use language_id as schema name for isolation
        try:
            index = DuckdbIndex(db_path, schema_name)
            log.info(f"Created DuckDB-based symbol index for {language_id} (schema: {schema_name}) at {db_path}")
            return index
        except ImportError as e:
            raise ImportError(
                "DuckDB is required for 'duckdb' backend. Install it with: pip install duckdb"
            ) from e
    else:
        raise ValueError(
            f"Unsupported backend type: {backend}. Supported backends: 'kuzu', 'duckdb'"
        )
