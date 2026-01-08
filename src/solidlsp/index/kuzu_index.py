"""
Kùzu-based graph database cache for symbol information.

This module provides an optimized symbol caching system using Kùzu graph database
to handle the hierarchical and circular nature of symbol trees more efficiently
than pickle-based caching.

Key features:
- Zero-copy graph traversal
- Automatic handling of circular references (parent-child relationships)
- Efficient incremental updates
- ACID transactions
- Rich Cypher queries for symbol lookup
"""

import logging
import json
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import shutil
import hashlib
import threading

try:
    import kuzu
except ImportError:
    kuzu = None

from solidlsp import ls_types
from .symbol_index import SymbolIndex,SymbolAppender,SingleAppender

log = logging.getLogger(__name__)


class KuzuIndex(SymbolIndex):
    """
    Graph-based symbol cache using Kùzu database.
    
    Implements SymbolIndex interface with Kùzu as the storage backend.
    Stores doc symbols as nodes and their relationships as edges,
    enabling efficient queries and automatic circular reference handling.
    """
    
    # Schema version for migration support
    SCHEMA_VERSION = 2  # v2: Path normalization (always use forward slashes)
    
    def __init__(self, base_dir: str, db_name: str="symbol_db"):
        """
        Initialize Kùzu symbol cache.
        
        :param base_dir: Directory to store Kùzu database files
        """        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_name = db_name
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Lazy initialization - db and conn will be created in start()
        self.db = None
        self.conn = None
    
    def is_started(self) -> bool:
        """Check if the database connection is started."""
        with self._lock:
            return self.conn is not None

    def start(self) -> None:
        """Start the database connection and initialize schema."""
        with self._lock:
            if self.conn is not None:
                return  # Already started
            
            db_path = self.base_dir / self.db_name
            self.db = kuzu.Database(str(db_path))
            self.conn = kuzu.Connection(self.db)
            
            # Initialize schema if needed
            self._init_schema()
            
            log.info(f"Started Kùzu symbol cache at {db_path}")
    
    def stop(self) -> None:
        """Stop the database connection and release resources."""
        with self._lock:
            if self.conn is None:
                return  # Already stopped
            
            # Kùzu Python bindings don't have explicit close()
            # Setting to None allows Python GC to clean up
            self.conn = None
            self.db = None
    
    def _init_schema(self) -> None:
        """Initialize the graph database schema."""
        with self._lock:
            try:
                # Create Doc node table
                self.conn.execute("""
                    CREATE NODE TABLE IF NOT EXISTS Doc (
                        id STRING PRIMARY KEY,
                        name STRING,
                        relative_path STRING,
                        content_hash STRING,
                        last_modified TIMESTAMP,
                        schema_version INT
                    )
                """)
                
                # Create Symbol node table
                self.conn.execute("""
                    CREATE NODE TABLE IF NOT EXISTS Symbol (
                        id STRING PRIMARY KEY,
                        name STRING,
                        kind INT,
                        start_line INT,
                        start_char INT,
                        end_line INT,
                        end_char INT,
                        body STRING,
                        overload_idx INT
                    )
                """)
                
                # Create DEFINES relationship (Doc -> Symbol)
                self.conn.execute("""
                    CREATE REL TABLE IF NOT EXISTS DEFINES (
                        FROM Doc TO Symbol
                    )
                """)
                
                # Create PART_OF relationship (Doc -> Symbol)
                self.conn.execute("""
                    CREATE REL TABLE IF NOT EXISTS PART_OF (
                        FROM Doc TO Symbol
                    )
                """)
                
                # Create PARENT_OF relationship (Symbol -> Symbol for hierarchy)
                self.conn.execute("""
                    CREATE REL TABLE IF NOT EXISTS PARENT_OF (
                        FROM Symbol TO Symbol
                    )
                """)
                
                log.debug("Kùzu schema initialized successfully")
                
            except Exception as e:
                # Schema might already exist, which is fine
                if "already exists" not in str(e):
                    log.warning(f"Schema initialization note: {e}")
    
    def store_doc_symbols(
        self,
        relative_path: str,
        content_hash: str,
        root_symbols: List[ls_types.UnifiedSymbolInformation]
    ) -> None:
        """
        Store doc symbols in the graph database.
        
        :param relative_path: Relative path of the doc
        :param content_hash: MD5 hash of doc content
        :param root_symbols: List of root-level symbols
        """
        with self._lock:
            doc_id = self._make_doc_id(relative_path)
            
            # Delete old symbols if doc exists (separate transaction)
            self._delete_doc_symbols(doc_id)
            
            # Begin transaction for atomic insertion
            self.conn.execute("BEGIN TRANSACTION")
            
            try:
                # Create doc node
                doc_name = Path(relative_path).name
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Normalize path separators for consistent querying
                normalized_relative_path = relative_path.replace('\\', '/')
                
                # Use CAST to convert string to TIMESTAMP
                self.conn.execute(
                    "CREATE (d:Doc {id: $id, name: $name, relative_path: $path, content_hash: $hash, last_modified: CAST($timestamp, 'TIMESTAMP'), schema_version: $version})",
                    {"id": doc_id, "name": doc_name, "path": normalized_relative_path, "hash": content_hash, "timestamp": timestamp_str, "version": self.SCHEMA_VERSION}
                )
                
                # Insert symbols recursively
                self._insert_symbols_recursive(
                    doc_id,
                    root_symbols,
                    parent_symbol_id=None
                )
                
                # Commit transaction
                self.conn.execute("COMMIT")
                log.debug(f"Stored {len(root_symbols)} root symbols for {relative_path}")
            except Exception as e:
                # Rollback on error
                self.conn.execute("ROLLBACK")
                log.error(f"Failed to store doc symbols for {relative_path}: {e}")
                raise
    
    def get_doc_symbols(
        self,
        relative_path: str,
        content_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached doc symbols if cache is valid.
        
        :param relative_path: Relative path of the doc
        :param content_hash: MD5 hash of current doc content
        :return: Dictionary with root_symbols and metadata, or None if cache invalid
        """
        with self._lock:
            doc_id = self._make_doc_id(relative_path)
            
            # Get doc info with cache validation
            result = self.conn.execute(
                "MATCH (d:Doc {id: $doc_id}) RETURN d.name, d.relative_path, d.content_hash",
                {"doc_id": doc_id}
            )
            
            if not result.has_next():
                log.debug(f"Doc not found in cache: {relative_path}")
                return None
            
            row = result.get_next()
            doc_name = row[0]
            doc_relative_path = row[1]
            stored_hash = row[2]
            
            # Validate cache
            if stored_hash != content_hash:
                log.debug(f"Cache for {relative_path} is invalid (hash mismatch)")
                return None
            
            # Get all root symbols (directly defined by doc)
            result = self.conn.execute(
                "MATCH (d:Doc {id: $doc_id})-[:DEFINES]->(s:Symbol) RETURN s.id",
                {"doc_id": doc_id}
            )
            
            # Build Doc symbol with children
            doc_symbol = {
                'id': doc_id,
                'name': doc_name,
                'kind': 1,  # File kind
                'relativePath': doc_relative_path,
                'children': []
            }
            
            # Build symbol trees as children of doc
            children = []
            while result.has_next():
                row = result.get_next()
                symbol_dict = self._build_symbol_tree(row[0], parent_ref=doc_symbol)
                if symbol_dict:
                    children.append(symbol_dict)

            doc_symbol['children'] = children
            return {
                'file_symbol': doc_symbol,
                'content_hash': content_hash
            }

    
    def query_symbols(
        self,
        name_path_pattern: str,
        substring_matching: bool = False,
        include_kinds: Optional[List[int]] = None,
        exclude_kinds: Optional[List[int]] = None,
        within_relative_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query symbols using Cypher to match name path patterns directly in the graph database.
        
        This method leverages Kùzu's graph traversal capabilities to efficiently find symbols
        based on their hierarchical path, avoiding the need to load entire symbol trees into memory.
        
        NOTE: within_relative_path should be a directory path or None. Doc paths should be
        handled by the caller using request_document_symbols() directly.
        
        :param name_path_pattern: Pattern to match symbol paths:
            - Empty string: "" matches all symbols (when used with within_relative_path)
            - Simple name: "method" matches any symbol named "method"
            - Relative path: "class/method" matches method under class
            - Absolute path: "/class/method" matches exact path from root
            - Overload index: "class/method[1]" matches specific overload
        :param substring_matching: Whether to use substring matching for the last segment
        :param include_kinds: List of symbol kinds to include (e.g., [12] for methods)
        :param exclude_kinds: List of symbol kinds to exclude
        :param within_relative_path: Limit search to directory (NOT doc - docs should be handled separately)
        :return: List of symbol dictionaries with full UnifiedSymbolInformation structure
        """
        with self._lock:
            # Parse the name path pattern
            is_absolute = name_path_pattern.startswith('/')
            pattern = name_path_pattern.lstrip('/').rstrip('/')
            
            # Handle empty pattern - split will give ['']
            parts = pattern.split('/') if pattern else ['']
            
            # Extract overload index if present
            overload_idx = None
            if parts[-1].endswith(']') and '[' in parts[-1]:
                last_part = parts[-1]
                idx_start = last_part.rfind('[')
                try:
                    overload_idx = int(last_part[idx_start+1:-1])
                    parts[-1] = last_part[:idx_start]
                except (ValueError, IndexError):
                    pass
            
            # Build Cypher query based on pattern structure
            if len(parts) == 1:
                # Simple name match: match any symbol with this name
                cypher_query = self._build_simple_name_query(
                    parts[0], substring_matching, include_kinds, exclude_kinds,
                    within_relative_path, overload_idx
                )
            else:
                # Path match: use graph traversal to match parent-child relationships
                cypher_query = self._build_path_match_query(
                    parts, is_absolute, substring_matching, include_kinds,
                    exclude_kinds, within_relative_path, overload_idx
                )
            
            log.debug(f"Executing Cypher query: {cypher_query}")
            import time
            start_time = time.perf_counter()
            result = self.conn.execute(cypher_query)
            
            # Collect matching symbol IDs
            symbol_ids = []
            while result.has_next():
                row = result.get_next()
                symbol_ids.append(row[0])  # symbol_id is first column
            
            query_time = time.perf_counter() - start_time
            
            # Build Doc symbols grouped by doc with matching symbols as children
            group_start_time = time.perf_counter()
            doc_symbols = self._group_symbols_by_doc(symbol_ids)
            group_time = time.perf_counter() - group_start_time
            
            total_time = time.perf_counter() - start_time
            log.debug(f"Found {len(doc_symbols)} docs with symbols matching pattern '{name_path_pattern}' (query: {query_time:.3f}s, group: {group_time:.3f}s, total: {total_time:.3f}s)")
            return doc_symbols
    
    def _build_simple_name_query(
        self,
        name: str,
        substring_matching: bool,
        include_kinds: Optional[List[int]],
        exclude_kinds: Optional[List[int]],
        within_relative_path: Optional[str],
        overload_idx: Optional[int]
    ) -> str:
        """Build Cypher query for simple name matching."""
        conditions = []
        
        # Name matching condition (skip if empty string)
        if name:
            if substring_matching:
                conditions.append(f"s.name CONTAINS '{name}'")
            else:
                conditions.append(f"s.name = '{name}'")
        
        # Kind filtering
        if include_kinds:
            kind_list = ', '.join(str(k) for k in include_kinds)
            conditions.append(f"s.kind IN [{kind_list}]")
        if exclude_kinds:
            kind_list = ', '.join(str(k) for k in exclude_kinds)
            conditions.append(f"NOT s.kind IN [{kind_list}]")
        
        # Overload index matching
        if overload_idx is not None:
            conditions.append(f"s.overload_idx = {overload_idx}")
        
        # Directory path filtering (doc paths should be handled separately by caller)
        doc_condition = ""
        if within_relative_path:
            # Normalize path separators
            normalized_path = within_relative_path.replace('\\', '/')
            # Always use prefix match for directories
            doc_condition = f" AND d.relative_path STARTS WITH '{normalized_path}'"
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        return f"""
            MATCH (d:Doc)-[:PART_OF]->(s:Symbol)
            WHERE {where_clause}{doc_condition}
            RETURN s.id AS symbol_id
        """
    
    def _build_path_match_query(
        self,
        parts: List[str],
        is_absolute: bool,
        substring_matching: bool,
        include_kinds: Optional[List[int]],
        exclude_kinds: Optional[List[int]],
        within_relative_path: Optional[str],
        overload_idx: Optional[int]
    ) -> str:
        """
        Build Cypher query for hierarchical path matching using graph traversal.
        
        Optimized to use PART_OF for faster deep symbol lookup and properly 
        validate all intermediate nodes in multi-level paths.
        """
        depth = len(parts)
        
        # Build target symbol conditions (last part of path)
        target_conditions = []
        if substring_matching:
            target_conditions.append(f"target.name CONTAINS '{parts[-1]}'")
        else:
            target_conditions.append(f"target.name = '{parts[-1]}'")
        
        # Kind filtering on target
        if include_kinds:
            kind_list = ', '.join(str(k) for k in include_kinds)
            target_conditions.append(f"target.kind IN [{kind_list}]")
        if exclude_kinds:
            kind_list = ', '.join(str(k) for k in exclude_kinds)
            target_conditions.append(f"NOT target.kind IN [{kind_list}]")
        
        # Overload index matching
        if overload_idx is not None:
            target_conditions.append(f"target.overload_idx = {overload_idx}")
        
        target_where = " AND ".join(target_conditions)
        
        # Directory path filtering
        doc_condition = ""
        if within_relative_path:
            normalized_path = within_relative_path.replace('\\', '/')
            doc_condition = f" AND d.relative_path STARTS WITH '{normalized_path}'"
        
        if depth == 2:
            # Two-level path: parent -> target
            # Use PART_OF for faster lookup, then verify parent relationship
            parent_name = parts[0]
            
            if is_absolute:
                # Absolute path: ensure parent is root (no grandparent)
                query = f"""
                    MATCH (d:Doc)-[:PART_OF]->(target:Symbol)
                    WHERE {target_where}{doc_condition}
                    MATCH (parent:Symbol)-[:PARENT_OF]->(target)
                    WHERE parent.name = '{parent_name}'
                        AND (d)-[:DEFINES]->(parent)
                    RETURN target.id AS symbol_id
                """
            else:
                # Relative path: parent can be anywhere
                query = f"""
                    MATCH (d:Doc)-[:PART_OF]->(target:Symbol)
                    WHERE {target_where}{doc_condition}
                    MATCH (parent:Symbol)-[:PARENT_OF]->(target)
                    WHERE parent.name = '{parent_name}'
                    RETURN target.id AS symbol_id
                """
        else:
            # Multi-level path (depth > 2): root -> ... -> target
            # Use PART_OF for target, then trace back and validate all intermediate names
            
            # Build path validation: check that all nodes in path match expected names
            # Path nodes order: [root, node1, node2, ..., target]
            # parts order: [parts[0], parts[1], parts[2], ..., parts[-1]]
            
            if is_absolute:
                # Absolute path: root must be directly defined by Doc
                query = f"""
                    MATCH (d:Doc)-[:PART_OF]->(target:Symbol)
                    WHERE {target_where}{doc_condition}
                    MATCH path = (root:Symbol)-[:PARENT_OF*{depth-1}]->(target)
                    WHERE (d)-[:DEFINES]->(root)
                        AND size(nodes(path)) = {depth}
                        AND root.name = '{parts[0]}'
                """
            else:
                # Relative path: root can be anywhere in the tree
                query = f"""
                    MATCH (d:Doc)-[:PART_OF]->(target:Symbol)
                    WHERE {target_where}{doc_condition}
                    MATCH path = (root:Symbol)-[:PARENT_OF*{depth-1}]->(target)
                    WHERE size(nodes(path)) = {depth}
                        AND root.name = '{parts[0]}'
                """
            
            # Add validation for all intermediate node names
            # We need to check that nodes(path)[i].name = parts[i] for all i
            for i in range(1, depth - 1):
                # parts[i] corresponds to nodes(path)[i]
                # Note: In Cypher, nodes(path)[0] is root, nodes(path)[depth-1] is target
                query += f"\n AND nodes(path)[{i}].name = '{parts[i]}'"
            
            query += "\n RETURN target.id AS symbol_id"
        
        return query

    def invalidate_doc(self, relative_path: str) -> None:
        """
        Invalidate cache for a specific doc.
        
        :param relative_path: Relative path of the doc to invalidate
        """
        with self._lock:
            doc_id = self._make_doc_id(relative_path)
            self._delete_doc_symbols(doc_id)
            log.debug(f"Invalidated cache for {relative_path}")
    
    def clear_all(self) -> None:
        """Clear all symbols from the cache."""
        with self._lock:
            if not self.is_started():
                return
            self.conn.execute("MATCH (n) DETACH DELETE n")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        with self._lock:
            doc_result = self.conn.execute(
                "MATCH (d:Doc) RETURN COUNT(*) AS count"
            )
            doc_count = 0
            if doc_result.has_next():
                doc_count = doc_result.get_next()[0]
            
            symbol_result = self.conn.execute(
                "MATCH (s:Symbol) RETURN COUNT(*) AS count"
            )
            symbol_count = 0
            if symbol_result.has_next():
                symbol_count = symbol_result.get_next()[0]
            
            return {
                'doc_count': int(doc_count),
                'symbol_count': int(symbol_count),
                'schema_version': self.SCHEMA_VERSION
            }
    
    def get_all_cached_docs(self) -> List[str]:
        """
        Get a list of all docs currently in the cache.
        
        :return: List of relative paths of cached docs
        """
        with self._lock:
            result = self.conn.execute("""
                MATCH (d:Doc)
                RETURN d.relative_path AS path
            """)
            
            docs = []
            while result.has_next():
                row = result.get_next()
                docs.append(row[0])
            
            return docs
    
    def create_appender(self, batch_mode: bool = False) -> SymbolAppender:
        """
        创建符号追加器。
        
        :param batch_mode: 是否使用批量模式（Kuzu 暂不支持批量模式）
        :return: SingleAppender 实例
        
        注意：Kuzu 当前实现总是返回 SingleAppender（每次立即提交）。
        """
        return SingleAppender(self)
    
    # ======================== Private Helper Methods ========================
    
    def _make_doc_id(self, relative_path: str) -> str:
        """Generate a unique ID for a doc."""
        return hashlib.blake2b(f"doc://{relative_path.replace(chr(92), '/')}".encode("utf-8"), digest_size=16).hexdigest()  # normalize backslashes
    
    def _make_symbol_id(self, doc_id: str, parent_symbol_id: Optional[str], symbol_name: str, start_line: int = 0, start_char: int = 0, overload_idx: int = 0) -> str:
        """Generate a unique ID for a symbol using name, position, parent, and optional overload index."""
        # Build base ID string with parent context
        if parent_symbol_id:
            base_id = f"sym://{doc_id}::{parent_symbol_id}::{symbol_name}@{start_line}:{start_char}"
        else:
            base_id = f"sym://{doc_id}::{symbol_name}@{start_line}:{start_char}"
        
        # Add overload index if present
        if overload_idx > 0:
            base_id += f"#{overload_idx}"
        
        return hashlib.blake2b(base_id.encode("utf-8"), digest_size=16).hexdigest()
    
    def is_doc_cached(self, relative_path: str, content_hash: str) -> bool:
        """
        Check if a document's symbols are already cached and valid.
        
        :param relative_path: Relative path of the document
        :param content_hash: MD5 hash of the document content
        :return: True if cached and valid, False otherwise
        """
        with self._lock:
            if self.conn is None:
                return False
            
            doc_id = self._make_doc_id(relative_path)
            return self._is_cache_valid(doc_id, content_hash)
    
    def _is_cache_valid(self, doc_id: str, content_hash: str) -> bool:
        """Check if cached symbols are still valid."""
        result = self.conn.execute(
            "MATCH (d:Doc {id: $doc_id}) RETURN d.content_hash AS hash",
            {"doc_id": doc_id}
        )
        
        # Get result without pandas
        if result.has_next():
            row = result.get_next()
            stored_hash = row[0]  # hash
            return stored_hash == content_hash
        
        return False
    
    def _delete_doc_symbols(self, doc_id: str) -> None:
        """Delete all symbols and relationships for a doc."""
        self.conn.execute("BEGIN TRANSACTION")
        try:
            # Delete relationships and symbols
            self.conn.execute(
                "MATCH (d:Doc {id: $doc_id})-[:PART_OF]->(s:Symbol) DETACH DELETE s",
                {"doc_id": doc_id}
            )
            
            # Delete doc node
            self.conn.execute(
                "MATCH (d:Doc {id: $doc_id}) DELETE d",
                {"doc_id": doc_id}
            )
            self.conn.execute("COMMIT")
        except Exception as e:
            self.conn.execute("ROLLBACK")
            log.error(f"Failed to delete doc symbols for {doc_id}: {e}")
            raise
    
    def _insert_symbols_recursive(
        self,
        doc_id: str,
        symbols: List[ls_types.UnifiedSymbolInformation],
        parent_symbol_id: Optional[str] = None
    ) -> None:
        """Recursively insert symbols and their children into the database."""
        for symbol in symbols:
            # Extract symbol information
            location = symbol.get('location', {})
            range_info = symbol.get('range', location.get('range', {}))
            start = range_info.get('start', {'line': 0, 'character': 0})
            end = range_info.get('end', {'line': 0, 'character': 0})

            overload_idx = symbol.get('overload_idx', 0)
            start_line = start.get('line', 0)
            start_char = start.get('character', 0)
            symbol_id = self._make_symbol_id(doc_id, parent_symbol_id, symbol['name'], start_line, start_char, overload_idx)

            body = symbol.get('body', '')
            name = symbol['name']

            # Create symbol node with parameterized query
            self.conn.execute(
                "CREATE (s:Symbol {id: $id, name: $name, kind: $kind, start_line: $start_line, start_char: $start_char, end_line: $end_line, end_char: $end_char, body: $body, overload_idx: $overload_idx})",
                {
                    "id": symbol_id,
                    "name": name,
                    "kind": symbol.get('kind', 0),
                    "start_line": start.get('line', 0),
                    "start_char": start.get('character', 0),
                    "end_line": end.get('line', 0),
                    "end_char": end.get('character', 0),
                    "body": body,
                    "overload_idx": overload_idx
                }
            )

            # Create PART_OF relationship from doc to symbol
            self.conn.execute(
                "MATCH (d:Doc {id: $doc_id}), (s:Symbol {id: $symbol_id}) CREATE (d)-[:PART_OF]->(s)",
                {"doc_id": doc_id, "symbol_id": symbol_id}
            )

            # Create DEFINES relationship from doc to symbol(only for root symbols)
            if parent_symbol_id is None:
                self.conn.execute(
                    "MATCH (d:Doc {id: $doc_id}), (s:Symbol {id: $symbol_id}) CREATE (d)-[:DEFINES]->(s)",
                    {"doc_id": doc_id, "symbol_id": symbol_id}
                )
            if parent_symbol_id:
                self.conn.execute(
                    "MATCH (p:Symbol {id: $parent_id}), (c:Symbol {id: $child_id}) CREATE (p)-[:PARENT_OF]->(c)",
                    {"parent_id": parent_symbol_id, "child_id": symbol_id}
                )
            if 'children' in symbol and symbol['children']:
                self._insert_symbols_recursive(
                    doc_id,
                    symbol['children'],
                    symbol_id
                )
      
    
    def _group_symbols_by_doc(self, symbol_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Group symbol IDs by doc and return Doc symbols with complete path chains.
        
        Optimized approach using batch queries:
        1. Get docs and matched symbols in one query
        2. For each doc, fetch all PART_OF symbols and PARENT_OF relationships in batch
        3. Build tree in Python (fast in-memory operation)
        
        :param symbol_ids: List of symbol IDs to group
        :return: List of Doc symbol dictionaries with filtered trees
        """
        if not symbol_ids:
            return []
        
        import time
        start = time.perf_counter()
        
        # Build IN clause for batch query
        ids_clause = "', '".join(symbol_ids)
        
        log.debug(f"Querying {len(symbol_ids)} symbols from graph database")
        
        # Step 1: Get docs with their matched symbols
        result = self.conn.execute(f"""
            MATCH (d:Doc)-[:PART_OF]->(matched:Symbol)
            WHERE matched.id IN ['{ids_clause}']
            RETURN d.id AS doc_id,
                   d.name AS doc_name,
                   d.relative_path AS doc_path,
                   d.content_hash AS doc_hash,
                   COLLECT(DISTINCT matched.id) AS matched_ids
        """)
        
        docs_info = []
        while result.has_next():
            row = result.get_next()
            docs_info.append({
                'doc_id': row[0],
                'doc_name': row[1],
                'doc_path': row[2],
                'doc_hash': row[3],
                'matched_ids': set(row[4])
            })
        
        step1_time = time.perf_counter() - start
        log.debug(f"Step 1 (doc grouping): {step1_time:.3f}s for {len(docs_info)} docs")
        
        # Step 2: For each doc, batch fetch all symbols and relationships
        doc_symbols = []
        step2_start = time.perf_counter()
        
        for doc_info in docs_info:
            doc_id = doc_info['doc_id']
            matched_ids = doc_info['matched_ids']
            
            # Fetch all symbols in this doc with parent-child relationships
            symbols_result = self.conn.execute("""
                MATCH (d:Doc {id: $doc_id})-[:PART_OF]->(s:Symbol)
                OPTIONAL MATCH (s)-[:PARENT_OF]->(child:Symbol)
                RETURN s.id, s.name, s.kind, s.start_line, s.start_char, s.end_line, s.end_char, s.body, s.overload_idx,
                       COLLECT(DISTINCT child.id) AS child_ids
                ORDER BY s.start_line
            """, {"doc_id": doc_id})
            
            # Build symbol lookup and parent-child map
            symbol_map = {}
            children_map = {}
            root_ids = set()
            
            while symbols_result.has_next():
                row = symbols_result.get_next()
                symbol_id = row[0]
                child_ids_raw = row[9] if row[9] is not None else []
                child_ids = [cid for cid in child_ids_raw if cid]  # Filter out None values
                
                symbol_map[symbol_id] = {
                    'id': symbol_id,
                    'name': row[1],
                    'kind': row[2],
                    'range': {
                        'start': {'line': row[3], 'character': row[4]},
                        'end': {'line': row[5], 'character': row[6]}
                    },
                    'selectionRange': {
                        'start': {'line': row[3], 'character': row[4]},
                        'end': {'line': row[5], 'character': row[6]}
                    },
                    'location': {
                        'range': {
                            'start': {'line': row[3], 'character': row[4]},
                            'end': {'line': row[5], 'character': row[6]}
                        }
                    },
                    'body': row[7],
                    'overload_idx': row[8],
                    'children': []
                }
                
                children_map[symbol_id] = child_ids
                
                # Track root symbols (will be determined after building children_map)
                root_ids.add(symbol_id)
            
            # Remove non-root symbols from root_ids
            for parent_id, child_id_list in children_map.items():
                for child_id in child_id_list:
                    root_ids.discard(child_id)
            
            # Build tree recursively from roots, filtering to matched symbols
            def build_tree_filtered(sid: str) -> Optional[Dict[str, Any]]:
                symbol = symbol_map[sid].copy()
                child_ids = children_map.get(sid, [])
                
                # Recursively build children
                filtered_children = []
                for cid in child_ids:
                    if cid in symbol_map:
                        child_tree = build_tree_filtered(cid)
                        if child_tree:
                            filtered_children.append(child_tree)
                
                # Include this symbol if it's matched or has matched descendants
                if sid in matched_ids or filtered_children:
                    symbol['children'] = filtered_children
                    return symbol
                
                return None
            
            # Build filtered trees for all roots
            filtered_roots = []
            for root_id in sorted(root_ids):
                if root_id in symbol_map:
                    tree = build_tree_filtered(root_id)
                    if tree:
                        filtered_roots.append(tree)
            
            # Create Doc symbol
            doc_symbol = {
                'id': doc_id,
                'name': doc_info['doc_name'],
                'kind': 1,  # File kind
                'relativePath': doc_info['doc_path'],
                'contentHash': doc_info['doc_hash'],
                'children': filtered_roots
            }
            doc_symbols.append(doc_symbol)
        
        step2_time = time.perf_counter() - step2_start
        total_time = time.perf_counter() - start
        log.debug(f"Step 2 (tree building): {step2_time:.3f}s, total: {total_time:.3f}s")
        
        return doc_symbols
    
    def _merge_paths_to_tree(
        self, 
        paths: List[List[str]],
        node_cache: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge multiple symbol paths into a tree structure.
        
        :param paths: List of paths, where each path is [root_id, ..., leaf_id]
        :param node_cache: Pre-fetched node information
        :return: List of root symbol dictionaries with merged children
        """
        if not paths:
            return []
        
        # Build parent->children mapping
        children_map: Dict[str, set] = {}  # parent_id -> set of child IDs
        root_ids = set()
        
        for path in paths:
            if not path:
                continue

            root_ids.add(path[0])
            
            # Build parent-child relationships
            for i in range(len(path) - 1):
                parent_id = path[i]
                child_id = path[i + 1]
                
                if parent_id not in children_map:
                    children_map[parent_id] = set()
                children_map[parent_id].add(child_id)
        
        # Recursively build tree
        def build_tree(symbol_id: str) -> Dict[str, Any]:
            symbol = node_cache[symbol_id].copy()
            
            # Get children and recursively build their trees
            child_ids = children_map.get(symbol_id, set())
            children = [build_tree(cid) for cid in sorted(child_ids)]
            
            symbol['children'] = children
            return symbol
        
        # Build trees for all roots
        return [build_tree(rid) for rid in sorted(root_ids)]

    def _find_root_symbol(self, symbol_id: str) -> Optional[str]:
        """
        Find the root symbol (direct child of Doc) for a given symbol.
        
        :param symbol_id: ID of the symbol
        :return: ID of the root symbol, or None if not found
        """
        # Traverse up until we find a symbol that is directly defined by a Doc
        result = self.conn.execute(
            "MATCH (d:Doc)-[:DEFINES]->(root:Symbol) WHERE root.id = $symbol_id OR (root)-[:PARENT_OF*1..]->(s:Symbol {id: $symbol_id}) RETURN root.id LIMIT 1",
            {"symbol_id": symbol_id}
        )
        if result.has_next():
            return result.get_next()[0]
        return None
        

    def _build_symbol_tree(self, symbol_id: str, parent_ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recursively build a symbol tree from the database.
        
        :param symbol_id: ID of the symbol to build tree for
        :param parent_ref: Parent symbol reference to set
        :return: Symbol dictionary with children and parent reference
        """
        symbol = self._get_symbol_by_id(symbol_id)
        if not symbol:
            return {}
        
        # Note: Not setting parent reference to avoid circular references in JSON serialization
        
        # Get children
        result = self.conn.execute(
            "MATCH (s:Symbol {id: $symbol_id})-[:PARENT_OF]->(c:Symbol) RETURN c.id ORDER BY c.start_line",
            {"symbol_id": symbol_id}
        )
        
        children = []
        while result.has_next():
            row = result.get_next()
            child = self._build_symbol_tree(row[0], parent_ref=symbol)  # Pass current symbol as parent
            if child:
                children.append(child)
        
        symbol['children'] = children
        return symbol
      
    
    def _get_symbol_by_id(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a symbol by its ID with complete UnifiedSymbolInformation structure."""
        # Get symbol data
        result = self.conn.execute(
            "MATCH (s:Symbol {id: $symbol_id}) RETURN s.id, s.name, s.kind, s.start_line, s.start_char, s.end_line, s.end_char, s.body, s.overload_idx",
            {"symbol_id": symbol_id}
        )
        
        if not result.has_next():
            return None
        
        row = result.get_next()
        
        return {
            'id': row[0],
            'name': row[1],
            'kind': row[2],
            'range': {
                'start': {'line': row[3], 'character': row[4]},
                'end': {'line': row[5], 'character': row[6]}
            },
            'selectionRange': {
                'start': {'line': row[3], 'character': row[4]},
                'end': {'line': row[5], 'character': row[6]}
            },
            'location': {
                'range': {
                    'start': {'line': row[3], 'character': row[4]},
                    'end': {'line': row[5], 'character': row[6]}
                },
            },
            'body': row[7],
            'overload_idx': row[8]
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'conn'):
                self.conn = None
            if hasattr(self, 'db'):
                self.db = None
        except Exception as e:
            log.debug(f"Error during cleanup: {e}")


