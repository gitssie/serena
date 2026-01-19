"""
DuckDB-based symbol index implementation.

This module provides a DuckDB-backed implementation of the SymbolIndex interface,
using SQL schemas for data isolation and supporting multiple index types in a
single database file.
"""

import logging
import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import duckdb

from solidlsp import ls_types
from .symbol_index import SymbolIndex,SymbolAppender,SingleAppender

log = logging.getLogger(__name__)


class DuckdbIndex(SymbolIndex):
    """
    DuckDB-based symbol index with schema isolation.
    
    Implements SymbolIndex interface using DuckDB as the storage backend.
    Each instance operates within its own schema, allowing multiple index
    types to coexist in the same database file.
    
    Features:
    - Schema-based data isolation
    - ACID transactions
    - SQL-based queries
    - Automatic index creation for performance
    """
    
    # Schema version for migration support
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str, schema_name: str):
        """
        Initialize DuckDB index.
        
        :param db_path: Path to the DuckDB database file
        :param schema_name: Schema name for data isolation (e.g., "symbols", "refs")
        """
        if duckdb is None:
            raise ImportError("duckdb package is not installed. Please install it with: pip install duckdb")
        
        self.db_path = str(Path(db_path).resolve())
        self.schema_name = schema_name
        self._lock = threading.RLock()
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        
    def is_started(self) -> bool:
        """Check if the index is started."""
        with self._lock:
            return self.conn is not None
    
    def start(self) -> None:
        """Start the index connection and initialize schema."""
        with self._lock:
            if self.conn is not None:
                return  # Already started
            
            # Ensure directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to DuckDB (each instance gets its own connection)
            self.conn = duckdb.connect(self.db_path)
            
            # Optimize DuckDB settings for batch inserts
            #self._optimize_connection()
            
            # Initialize schema
            self._init_schema()
            
            log.debug(f"Started DuckDB index with schema '{self.schema_name}' at {self.db_path}")
    
    def stop(self) -> None:
        """Stop the index connection and release resources."""
        with self._lock:
            if self.conn is None:
                return  # Already stopped
            
            self.conn.close()
            self.conn = None
            log.debug(f"Stopped DuckDB index with schema '{self.schema_name}'")
    
    def _optimize_connection(self) -> None:
        """Optimize DuckDB connection settings for better batch insert performance."""
        try:
            # Increase memory limit (default is often too low for large datasets)
            # Set to 2GB or 50% of available memory, whichever is smaller
            self.conn.execute("SET memory_limit='2GB'")
            
            # Increase thread count for parallel processing
            # DuckDB will automatically use available CPU cores
            self.conn.execute("SET threads TO 4")
            
            # Enable parallel processing
            self.conn.execute("SET enable_object_cache=true")
            
            # Optimize for write-heavy workloads
            self.conn.execute("SET preserve_insertion_order=false")
            
            log.debug("DuckDB connection optimized for batch operations")
        except Exception as e:
            log.warning(f"Failed to optimize DuckDB settings: {e}")
    
    def _init_schema(self) -> None:
        """Initialize the schema and table structures."""
        with self._lock:
            log.debug(f"Starting schema initialization for '{self.schema_name}'")
            
            # Create schema
            log.debug(f"Creating schema '{self.schema_name}'...")
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            log.debug(f"Schema '{self.schema_name}' created successfully")
            
            # Create docs table
            log.debug(f"Creating docs table in schema '{self.schema_name}'...")
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.docs (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    relative_path VARCHAR,
                    content_hash VARCHAR,
                    last_modified TIMESTAMP,
                    schema_version INTEGER
                )
            """)
            log.debug(f"Docs table created successfully in schema '{self.schema_name}'")
            
            # Create symbols table
            log.debug(f"Creating symbols table in schema '{self.schema_name}'...")
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_name}.symbols (
                    id VARCHAR PRIMARY KEY,
                    doc_id VARCHAR NOT NULL,
                    parent_id VARCHAR,
                    parent_ids VARCHAR[],
                    name VARCHAR NOT NULL,
                    name_path VARCHAR NOT NULL,
                    kind INTEGER NOT NULL,
                    start_line INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    body TEXT,
                    overload_idx INTEGER DEFAULT 0,
                    FOREIGN KEY (doc_id) REFERENCES {self.schema_name}.docs(id)
                )
            """)
            log.debug(f"Symbols table created successfully in schema '{self.schema_name}'")
            
            # Create indexes for performance
            log.debug(f"Creating indexes for schema '{self.schema_name}'...")
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.schema_name}_symbols_doc 
                ON {self.schema_name}.symbols(doc_id)
            """)
            
            # self.conn.execute(f"""
            #     CREATE INDEX IF NOT EXISTS idx_{self.schema_name}_symbols_name 
            #     ON {self.schema_name}.symbols(name)
            # """)
            
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.schema_name}_symbols_parent 
                ON {self.schema_name}.symbols(parent_id)
            """)
            
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.schema_name}_symbols_name_path 
                ON {self.schema_name}.symbols(name_path)
            """)
            log.debug(f"Indexes created successfully for schema '{self.schema_name}'")
            
            log.debug(f"Initialized schema '{self.schema_name}' successfully")
    
    def store_doc_symbols(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """Store doc symbols in the database."""
        with self._lock:
            if not self.is_started():
                raise RuntimeError("Index not started. Call start() first.")
            
            try:
                # Create rows
                doc_row, symbols_rows, doc_id = self._create_symbols_rows(
                    relative_path, content_hash, root_symbols
                )
                
                # Batch save using unified method
                self._batch_save(
                    docs_rows=[doc_row],
                    symbols_rows=symbols_rows,
                    doc_ids_to_delete=[doc_id]
                )
                
                log.debug(f"Stored {len(root_symbols)} root symbols for {relative_path}")
                
            except Exception as e:
                log.error(f"Failed to store doc symbols for {relative_path}: {e}")
                raise
    
    def get_doc_symbols(
        self,
        relative_path: str,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached doc symbols if cache is valid."""
        with self._lock:
            if not self.is_started():
                raise RuntimeError("Index not started. Call start() first.")
            
            doc_id = self._make_doc_id(relative_path)
            
            # Get doc info with cache validation
            result = self.conn.execute(f"""
                SELECT name, relative_path, content_hash 
                FROM {self.schema_name}.docs 
                WHERE id = ?
            """, [doc_id]).fetchall()
            
            if not result:
                log.debug(f"Doc not found in cache: {relative_path}")
                return None
            
            doc_name, doc_relative_path, stored_hash = result[0]
            
            # Validate cache
            if stored_hash != content_hash:
                log.debug(f"Cache for {relative_path} is invalid (hash mismatch)")
                return None
            
            # Build doc symbol with children
            doc_symbol = {
                'id': doc_id,
                'name': doc_name,
                'kind': 1,  # File kind
                'relativePath': doc_relative_path,
                'children': []
            }
            
            # Get ALL symbols for this doc in one query
            all_symbols_relation = self.conn.execute(f"""
                SELECT id, parent_id, name, parent_ids, name_path, kind, 
                       start_line, start_char, end_line, end_char, body, overload_idx
                FROM {self.schema_name}.symbols 
                WHERE doc_id = ?
                ORDER BY start_line
            """, [doc_id])
            
            all_symbols_result = all_symbols_relation.fetchall()
            columns = [desc[0] for desc in all_symbols_relation.description]
            
            # Build symbol map
            symbol_map = {}
            for row in all_symbols_result:
                row_dict = self._row_to_dict(row, columns)
                symbol_data = self._build_symbol_data(row_dict, doc_id)
                symbol_id = symbol_data['id']
                symbol_map[symbol_id] = symbol_data
            
            # Build symbol lookup map and children map
            for symbol_data in symbol_map.values():
                doc_id = symbol_data['doc_id']
                parent_id = symbol_data['parent_id']
                
                if parent_id is None:
                    # Root symbol, add to doc's children
                    doc_symbol['children'].append(symbol_data)
                else:
                    # Attach to parent if exists
                    parent_symbol = symbol_map.get(parent_id)
                    if parent_symbol:
                        parent_symbol['children'].append(symbol_data)
            
            return {
                'file_symbol': doc_symbol,
                'content_hash': content_hash
            }
    
    def is_doc_cached(self, relative_path: str, content_hash: str) -> bool:
        """Check if a document's symbols are already cached and valid."""
        with self._lock:
            if not self.is_started():
                return False
            
            doc_id = self._make_doc_id(relative_path)
            
            result = self.conn.execute(f"""
                SELECT content_hash FROM {self.schema_name}.docs 
                WHERE id = ?
            """, [doc_id]).fetchall()
            
            if not result:
                return False
            
            stored_hash = result[0][0]
            return stored_hash == content_hash
    
    def query_symbols(
        self,
        name_path_regex: str,
        include_kinds: Optional[List[int]] = None,
        exclude_kinds: Optional[List[int]] = None,
        relative_path_regex: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query symbols using regex pattern matching.
        
        :param name_path_regex: Regular expression to match symbol name_path
        :param include_kinds: Optional list of symbol kinds to include
        :param exclude_kinds: Optional list of symbol kinds to exclude
        :param relative_path_regex: Optional regex to match file relative_path
        :return: List of document symbols matching the criteria
        """
        with self._lock:
            if not self.is_started():
                raise RuntimeError("Index not started. Call start() first.")
            
            # Build SQL query
            sql_query, params = self._build_symbol_query(
                name_path_regex, include_kinds,
                exclude_kinds, relative_path_regex
            )
            
            log.debug(f"Executing SQL query: {sql_query}")
            start_time = time.perf_counter()
            
            relation = self.conn.execute(sql_query, params)
            result = relation.fetchall()
            columns = [desc[0] for desc in relation.description]
            
            query_time = time.perf_counter() - start_time
            
            # Build Doc symbols grouped by doc from full result set
            group_start_time = time.perf_counter()
            doc_symbols = self._build_doc_symbols_from_results(result, columns)
            group_time = time.perf_counter() - group_start_time
            
            total_time = time.perf_counter() - start_time
            log.debug(f"Found {len(doc_symbols)} docs with symbols matching regex '{name_path_regex}' "
                     f"(query: {query_time:.3f}s, group: {group_time:.3f}s, total: {total_time:.3f}s)")
            
            return doc_symbols
    
    def invalidate_doc(self, relative_path: str) -> None:
        """Invalidate cache for a specific doc."""
        with self._lock:
            if not self.is_started():
                return
            
            doc_id = self._make_doc_id(relative_path)
            self._delete_doc_symbols(doc_id)
    
    def clear_all(self) -> None:
        """Clear all symbols from the cache."""
        with self._lock:
            if not self.is_started():
                return
            
            self.conn.execute(f"TRUNCATE TABLE {self.schema_name}.symbols")
            self.conn.execute(f"TRUNCATE TABLE {self.schema_name}.docs")
    
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        with self._lock:
            if not self.is_started():
                return {
                    'total_docs': 0,
                    'total_symbols': 0,
                    'backend': 'duckdb',
                    'schema': self.schema_name
                }
            
            doc_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM {self.schema_name}.docs
            """).fetchone()[0]
            
            symbol_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM {self.schema_name}.symbols
            """).fetchone()[0]
            
            return {
                'total_docs': doc_count,
                'total_symbols': symbol_count,
                'backend': 'duckdb',
                'schema': self.schema_name,
                'db_path': self.db_path
            }
    
    def get_all_cached_docs(self) -> List[str]:
        """Get a list of all docs currently in the cache."""
        with self._lock:
            if not self.is_started():
                return []
            
            result = self.conn.execute(f"""
                SELECT relative_path FROM {self.schema_name}.docs 
                ORDER BY relative_path
            """).fetchall()
            
            return [row[0] for row in result]
    
    def create_appender(self, batch_mode: bool = False) -> SymbolAppender:
        """
        创建符号追加器。
        
        :param batch_mode: 是否使用批量模式（True=BatchAppender，False=SingleAppender）
        :return: SymbolAppender 实例
        
        BatchAppender 会累积多个文档的数据并在 commit() 时批量提交，
        提高大规模索引操作的性能。
        """
        if batch_mode:
            return BatchAppender(self)
        else:
            return SingleAppender(self)
    
    # ======================== Private Helper Methods ========================
    
    def _row_to_dict(self, row: tuple, columns: List[str]) -> Dict[str, Any]:
        """Convert tuple row to dictionary using column names."""
        return dict(zip(columns, row))
    
    def _build_symbol_data(self, row_dict: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
        """Build symbol data from row dictionary."""
        # Use 'symbol_id' if present (from parent symbol queries), otherwise 'id'
        symbol_id = row_dict.get('symbol_id', row_dict.get('id'))
        return {
            'id': symbol_id,
            'doc_id': doc_id,
            'parent_id': row_dict['parent_id'],
            'name': row_dict.get('symbol_name', row_dict.get('name')),
            'parent_ids': row_dict['parent_ids'] if row_dict['parent_ids'] else [],
            'name_path': row_dict['name_path'],
            'kind': row_dict['kind'],
            'range': {
                'start': {'line': row_dict['start_line'], 'character': row_dict['start_char']},
                'end': {'line': row_dict['end_line'], 'character': row_dict['end_char']}
            },
            'selectionRange': {
                'start': {'line': row_dict['start_line'], 'character': row_dict['start_char']},
                'end': {'line': row_dict['end_line'], 'character': row_dict['end_char']}
            },
            'location': {
                'range': {
                    'start': {'line': row_dict['start_line'], 'character': row_dict['start_char']},
                    'end': {'line': row_dict['end_line'], 'character': row_dict['end_char']}
                },
            },
            'body': row_dict['body'],
            'overload_idx': row_dict['overload_idx'],
            'children': []
        }
    
    def _query_missing_parent_symbols(self, missing_parent_ids: set) -> tuple:
        """Query missing parent symbols from database."""
        if not missing_parent_ids:
            return [], []
        
        missing_ids_list = list(missing_parent_ids)
        placeholders = ','.join('?' * len(missing_ids_list))
        missing_relation = self.conn.execute(f"""
            SELECT s.id as symbol_id, s.doc_id, s.parent_id, s.parent_ids, s.name as symbol_name, 
                   s.name_path, s.kind, s.start_line, s.start_char, s.end_line, s.end_char, 
                   s.body, s.overload_idx
            FROM {self.schema_name}.symbols s
            WHERE s.id IN ({placeholders})
        """, missing_ids_list)
        
        missing_result = missing_relation.fetchall()
        missing_columns = [desc[0] for desc in missing_relation.description]
        
        return missing_result, missing_columns
    
    def _query_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Query doc info from database."""
        doc_result = self.conn.execute(f"""
            SELECT name, relative_path, content_hash
            FROM {self.schema_name}.docs
            WHERE id = ?
        """, [doc_id]).fetchall()
        
        if doc_result:
            doc_row = doc_result[0]
            return {
                'name': doc_row[0],
                'relative_path': doc_row[1],
                'content_hash': doc_row[2]
            }
        return None
    
    def _add_missing_parent_symbols(self, symbol_map: Dict[str, Any]) -> None:
        """Add missing parent symbols to symbol_map and doc_map."""
         # ===== Step 2: Collect missing parent IDs =====
        missing_parent_ids = set()
        for symbol_data in symbol_map.values():
            for parent_id in symbol_data['parent_ids']:
                if parent_id not in symbol_map:
                    missing_parent_ids.add(parent_id)
        
        if not missing_parent_ids:
            return

        missing_result, missing_columns = self._query_missing_parent_symbols(missing_parent_ids)
        
        for row in missing_result:
            row_dict = self._row_to_dict(row, missing_columns)
            doc_id = row_dict['doc_id']
            symbol_data = self._build_symbol_data(row_dict, doc_id)
            symbol_id = symbol_data['id']
            symbol_map[symbol_id] = symbol_data
    
    def _create_doc_row(self, relative_path: str, content_hash: str) -> tuple:
        """Create a doc row tuple for database insertion.
        
        :param relative_path: Relative path of the document
        :param content_hash: Content hash of the document
        :return: Tuple (doc_id, doc_name, normalized_path, content_hash, schema_version)
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        doc_id = self._make_doc_id(relative_path)
        doc_name = Path(relative_path).name
        normalized_path = relative_path.replace('\\', '/')
        return (doc_id, doc_name, normalized_path, content_hash, self.SCHEMA_VERSION)
    
    def _create_symbols_rows(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> tuple:
        """Create doc and symbols rows for database insertion.
        
        :param relative_path: Relative path of the document
        :param content_hash: Content hash of the document
        :param root_symbols: Root symbols of the document
        :return: Tuple (doc_row, symbols_rows, doc_id)
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        doc_row = self._create_doc_row(relative_path, content_hash)
        doc_id = doc_row[0]
        symbols_rows = self._collect_symbols_recursive(
            doc_id=doc_id,
            symbols=root_symbols,
            parent_symbol_id=None,
            parent_path="",
            parent_ids_chain=None
        )
        return doc_row, symbols_rows, doc_id
    
    def _create_doc_row(self, relative_path: str, content_hash: str) -> tuple:
        """Create a doc row tuple for database insertion.
        
        :param relative_path: Relative path of the document
        :param content_hash: Content hash of the document
        :return: Tuple (doc_id, doc_name, normalized_path, content_hash, schema_version)
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        doc_id = self._make_doc_id(relative_path)
        doc_name = Path(relative_path).name
        normalized_path = relative_path.replace('\\', '/')
        return (doc_id, doc_name, normalized_path, content_hash, self.SCHEMA_VERSION)
    
    def _create_symbols_rows(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> tuple:
        """Create doc and symbols rows for database insertion.
        
        :param relative_path: Relative path of the document
        :param content_hash: Content hash of the document
        :param root_symbols: Root symbols of the document
        :return: Tuple (doc_row, symbols_rows, doc_id)
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        doc_row = self._create_doc_row(relative_path, content_hash)
        doc_id = doc_row[0]
        symbols_rows = self._collect_symbols_recursive(
            doc_id=doc_id,
            symbols=root_symbols,
            parent_symbol_id=None,
            parent_path="",
            parent_ids_chain=None
        )
        return doc_row, symbols_rows, doc_id
    
    def _make_doc_id(self, relative_path: str) -> str:
        """Generate a unique ID for a doc.
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        normalized_path = relative_path.replace('\\', '/')
        return hashlib.blake2b(f"doc://{normalized_path}".encode("utf-8"), digest_size=16).hexdigest()
    
    def _make_symbol_id(
        self,
        doc_id: str,
        parent_symbol_id: Optional[str],
        symbol_name: str,
        start_line: int = 0,
        start_char: int = 0,
        overload_idx: int = 0
    ) -> str:
        """Generate a unique ID for a symbol."""
        if parent_symbol_id:
            base_id = f"{doc_id}::{parent_symbol_id}::{symbol_name}::{start_line}:{start_char}"
        else:
            base_id = f"{doc_id}::{symbol_name}::{start_line}:{start_char}"
        
        if overload_idx > 0:
            base_id += f"::[{overload_idx}]"
        
        return hashlib.blake2b(base_id.encode("utf-8"), digest_size=16).hexdigest()
    
    def _batch_save(
        self,
        docs_rows: List[tuple],
        symbols_rows: List[tuple],
        doc_ids_to_delete: List[str]
    ) -> None:
        """Unified batch save method for docs and symbols.
        
        :param docs_rows: List of doc row tuples
        :param symbols_rows: List of symbol row tuples
        :param doc_ids_to_delete: List of doc IDs to delete before insertion
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        # Delete existing docs and symbols
        if doc_ids_to_delete:
            placeholders = ','.join('?' * len(doc_ids_to_delete))
            self.conn.execute(
                f"DELETE FROM {self.schema_name}.symbols WHERE doc_id IN ({placeholders})",
                doc_ids_to_delete
            )
            self.conn.execute(
                f"DELETE FROM {self.schema_name}.docs WHERE id IN ({placeholders})",
                doc_ids_to_delete
            )
        
        # Batch insert docs
        if docs_rows:
            self.conn.executemany(f"""
                INSERT INTO {self.schema_name}.docs 
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, docs_rows)
        
        # Batch insert symbols
        if symbols_rows:
            self.conn.executemany(f"""
                INSERT INTO {self.schema_name}.symbols 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, symbols_rows)
    
    def _delete_doc_symbols(self, doc_id: str) -> None:
        """Delete all symbols and doc for a doc_id."""
        # DuckDB doesn't support CASCADE, so delete symbols first
        self.conn.execute(f"""
            DELETE FROM {self.schema_name}.symbols WHERE doc_id = ?
        """, [doc_id])
        # Then delete the doc
        self.conn.execute(f"""
            DELETE FROM {self.schema_name}.docs WHERE id = ?
        """, [doc_id])
    
    def _collect_symbols_recursive(
        self,
        doc_id: str,
        symbols: List[ls_types.UnifiedSymbolInformation],
        parent_symbol_id: Optional[str],
        parent_path: str = "",
        parent_ids_chain: Optional[List[str]] = None,
        collected_rows: Optional[List[tuple]] = None
    ) -> List[tuple]:
        """Recursively collect symbol data for batch insertion.
        
        Note: This method is exposed as public API for use by SymbolAppender.
        """
        if parent_ids_chain is None:
            parent_ids_chain = []
        if collected_rows is None:
            collected_rows = []
        
        for symbol in symbols:
            # Extract symbol information
            symbol_name = symbol['name']
            location = symbol.get('location', {})
            range_info = symbol.get('range', location.get('range', {}))
            start = range_info.get('start', {'line': 0, 'character': 0})
            end = range_info.get('end', {'line': 0, 'character': 0})
            
            overload_idx = symbol.get('overload_idx', 0)
            start_line = start.get('line', 0)
            start_char = start.get('character', 0)
            symbol_id = self._make_symbol_id(doc_id, parent_symbol_id, symbol_name, 
                                            start_line, start_char, overload_idx)
            
            # Build symbol path with overload index if needed
            if parent_path:
                symbol_path = f"{parent_path}/{symbol_name}"
            else:
                symbol_path = symbol_name
            
            # Append overload index using # symbol (regex-safe)
            if overload_idx > 0:
                symbol_path += f"#{overload_idx}"
            
            # Build parent IDs chain for this symbol
            current_parent_ids = parent_ids_chain.copy()
            if parent_symbol_id:
                current_parent_ids.append(parent_symbol_id)
            
            # Collect symbol data as tuple
            row = (
                symbol_id,
                doc_id,
                parent_symbol_id,
                current_parent_ids,
                symbol_name,
                symbol_path,
                symbol.get('kind', 0),
                start.get('line', 0),
                start.get('character', 0),
                end.get('line', 0),
                end.get('character', 0),
                symbol.get('body', None),
                overload_idx
            )
            collected_rows.append(row)
            
            # Recursively collect children with updated chain
            if 'children' in symbol and symbol['children']:
                self._collect_symbols_recursive(doc_id, symbol['children'], symbol_id, 
                                               symbol_path, current_parent_ids, collected_rows)
        
        return collected_rows
    
    def _build_symbol_tree(self, symbol_id: str, parent_ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Recursively build a symbol tree from the database."""
        symbol = self._get_symbol_by_id(symbol_id)
        if not symbol:
            return None
        
        # Get children
        children_result = self.conn.execute(f"""
            SELECT id FROM {self.schema_name}.symbols 
            WHERE parent_id = ?
            ORDER BY start_line
        """, [symbol_id]).fetchall()
        
        children = []
        for (child_id,) in children_result:
            child_dict = self._build_symbol_tree(child_id, parent_ref=symbol)
            if child_dict:
                children.append(child_dict)
        
        symbol['children'] = children
        return symbol
    
    def _get_symbol_by_id(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a symbol by its ID."""
        relation = self.conn.execute(f"""
            SELECT id, doc_id, parent_id, parent_ids, name, name_path, kind, 
                   start_line, start_char, end_line, end_char, body, overload_idx
            FROM {self.schema_name}.symbols 
            WHERE id = ?
        """, [symbol_id])
        
        result = relation.fetchall()
        if not result:
            return None
        
        columns = [desc[0] for desc in relation.description]
        row_dict = self._row_to_dict(result[0], columns)
        
        return self._build_symbol_data(row_dict, row_dict['doc_id'])
    
    def _build_symbol_query(
        self,
        name_path_regex: str,
        include_kinds: Optional[List[int]],
        exclude_kinds: Optional[List[int]],
        relative_path_regex: Optional[str]
    ) -> tuple:
        """
        Build unified SQL query for symbol matching using regex.
        All pattern matching is done via regexp_matches() for consistency.
        
        :param name_path_regex: Regular expression to match symbol name_path
        :param include_kinds: Optional list of symbol kinds to include
        :param exclude_kinds: Optional list of symbol kinds to exclude  
        :param relative_path_regex: Optional regex to match file relative_path
        :return: (sql_query, params) tuple
        """
        conditions = []
        params = []
        
        # Use the provided regex pattern directly
        conditions.append("regexp_matches(s.name_path, ?)")
        params.append(name_path_regex)
        
        # Kind filtering
        if include_kinds:
            placeholders = ','.join('?' * len(include_kinds))
            conditions.append(f"s.kind IN ({placeholders})")
            params.extend(include_kinds)
        
        if exclude_kinds:
            placeholders = ','.join('?' * len(exclude_kinds))
            conditions.append(f"s.kind NOT IN ({placeholders})")
            params.extend(exclude_kinds)
        
        # Directory path filtering (direct regex matching)
        if relative_path_regex:
            conditions.append("regexp_matches(d.relative_path, ?)")
            params.append(relative_path_regex)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT s.id, s.doc_id, s.parent_id, s.parent_ids, s.name, 
                   s.name_path, s.kind, s.start_line, s.start_char, s.end_line, s.end_char, 
                   s.body, s.overload_idx, d.name as doc_name, d.relative_path, d.content_hash
            FROM {self.schema_name}.symbols s
            JOIN {self.schema_name}.docs d ON s.doc_id = d.id
            WHERE {where_clause}
            ORDER BY d.id, s.kind
        """
        
        return query, params
    
    def _build_doc_symbols_from_results(self, results: List[tuple], columns: List[str]) -> List[Dict[str, Any]]:
        """Build doc symbols from query results containing full symbol data."""
        if not results:
            return []
        
        # ===== Step 1: Group by symbol_id first =====
        symbol_map = {}  # {symbol_id: symbol_data}
        doc_map = {}     # {doc_id: doc_data}
        
        for row in results:
            row_dict = self._row_to_dict(row, columns)
            
            # Extract doc info
            doc_id = row_dict['doc_id']
            if doc_id not in doc_map:
                doc_map[doc_id] =  {
                    'id': doc_id,
                    'name': row_dict['doc_name'],
                    'kind': 1,
                    'relativePath': row_dict['relative_path'],
                    'children': []
                }
            # Extract symbol info (without doc-specific fields)
            symbol_id = row_dict['id']
            symbol_data = self._build_symbol_data(row_dict, doc_id)
            symbol_map[symbol_id] = symbol_data
        
        # ===== Step 2: Add missing parent symbols =====
        self._add_missing_parent_symbols(symbol_map)

        # ===== Step 3: Build doc symbols with tree structure =====
        for symbol_data in symbol_map.values():
            doc_id = symbol_data['doc_id']
            parent_id = symbol_data['parent_id']
            
            if parent_id is None:
                # Root symbol, add to doc's children
                doc_map[doc_id]['children'].append(symbol_data)
            else:
                # Attach to parent if exists
                parent_symbol = symbol_map.get(parent_id)
                if parent_symbol:
                    parent_symbol['children'].append(symbol_data)
        
        return list(doc_map.values())
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception as e:
            log.warning(f"Error closing DuckDB connection in destructor: {e}")


class BatchAppender(SymbolAppender):
    """
    DuckDB 专用的批量追加器。
    
    累积格式化后的 doc rows 和 symbol rows，当数据量达到 batch_size 时自动批量提交。
    使用 _create_symbols_rows 格式化数据，使用 _batch_save 批量入库。
    """
    
    def __init__(self, duckdb_index: 'DuckdbIndex'):
        """
        初始化批量追加器。
        
        :param duckdb_index: DuckdbIndex 实例
        """
        self.index = duckdb_index
        self.docs_rows = []  # type: List[tuple]
        self.symbols_rows = []  # type: List[tuple]
        self.doc_ids_to_delete = []  # type: List[str]
        self.batch_size = 3000  # 以行数为单位的批量大小
    
    def append(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """格式化并累积数据。当 docs + symbols 总行数达到 batch_size 时自动提交。"""
        # 使用 _create_symbols_rows 格式化数据
        doc_row, symbols_rows, doc_id = self.index._create_symbols_rows(
            relative_path, content_hash, root_symbols
        )
        
        # 累积数据
        self.docs_rows.append(doc_row)
        self.symbols_rows.extend(symbols_rows)
        self.doc_ids_to_delete.append(doc_id)
        
        # 当总行数达到 batch_size 时，自动进行批量提交
        total_rows = len(self.docs_rows) + len(self.symbols_rows)
        if total_rows >= self.batch_size:
            self._flush()
    
    def _flush(self) -> None:
        """内部方法：批量提交当前累积的数据。"""
        if not self.docs_rows:
            return
        
        # 使用统一的 _batch_save 方法
        self.index._batch_save(
            docs_rows=self.docs_rows,
            symbols_rows=self.symbols_rows,
            doc_ids_to_delete=self.doc_ids_to_delete
        )
        
        # 清空缓存
        self.docs_rows.clear()
        self.symbols_rows.clear()
        self.doc_ids_to_delete.clear()
    
    def commit(self) -> None:
        """批量提交剩余的缓存数据。"""
        self._flush()